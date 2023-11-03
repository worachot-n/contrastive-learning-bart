import json
import csv
import random
from random import randint
import warnings
import numpy as np
from collections.abc import Mapping
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import nltk
from nltk.corpus import wordnet
import datasets
from datasets import Dataset
import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy

import utils
from special_token import simple_tokenize, lemmatize_text, build_tagger


def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return synonyms


def load_from_dialogsum(args, file_path, split_type=None):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    id_list = [sample['fname'] for sample in data]
    dialogue_list = [sample['dialogue'] for sample in data]

    if 'summary' in data[0]:
        # summary
        summary_list = [sample['summary'] for sample in data]
        # topic
        topic_list = [sample['topic'] for sample in data]

    elif 'summary1' in data[0]:
        # id
        id_list1 = [id+"_sum1" for id in id_list]
        id_list2 = [id+"_sum2" for id in id_list]
        id_list3 = [id+"_sum3" for id in id_list]
        # all id and dialogue
        id_list = id_list1 + id_list2 + id_list3
        dialogue_list = dialogue_list + dialogue_list + dialogue_list
        # summary
        summary_list1 = [sample['summary1'] for sample in data]
        summary_list2 = [sample['summary2'] for sample in data]
        summary_list3 = [sample['summary3'] for sample in data]
        # all summary
        summary_list = summary_list1 + summary_list2 + summary_list3
        # topic
        topic_list1 = [sample['topic1'] for sample in data]
        topic_list2 = [sample['topic2'] for sample in data]
        topic_list3 = [sample['topic3'] for sample in data]
        # all topic
        topic_list = topic_list1 + topic_list2 + topic_list3

    data_dict = {'id': id_list,
             'dialogue': dialogue_list,
             'summary': summary_list,
             'topic': topic_list}
    
    if args.contrastive_loss:
        if split_type == 'train':
            topic_set = set(topic_list)
            positive_topic_list = []
            negative_topic_list = []
            for topic in topic_list:
                if args.postive_gen:
                    tokenized_text = nltk.word_tokenize(topic)
                    positive_topic = []
                    for word in tokenized_text:
                        if word not in {'a', 'an'}:
                            synonyms = get_synonyms(word)
                            synonyms_not_duplicate = set(synonyms).difference(set([word]))
                            if len(synonyms_not_duplicate):
                                synonyms_not_duplicate = random.choice(list(synonyms_not_duplicate))
                            else:
                                synonyms_not_duplicate = word
                            positive_topic.append(synonyms_not_duplicate)
                    positive_topic_list.append(' '.join(positive_topic))
                if args.negative_gen:
                    random_topic = topic_set.difference(set([topic]))
                    negative_topic = random.sample(list(random_topic), args.negative_sample)
                    negative_topic_list.append(negative_topic)
                    data_dict['negative_topic'] = negative_topic_list
            if args.postive_gen:
                data_dict['positive_topic'] = positive_topic_list
            if args.negative_gen:
                data_dict['negative_topic'] = negative_topic_list
        else:
            if args.postive_gen:
                # data_dict['positive_topic'] = topic_list
                data_dict['positive_topic'] = []
            if args.negative_gen:
                # data_dict['negative_topic'] = topic_list
                data_dict['negative_topic'] = []

    if args.topic_tagger:
        if split_type == 'train':
            topic_tagger = []
            positive_topic_tagger = []
            negative_topic_tagger = []
            original_tokens = [simple_tokenize(x) for x in dialogue_list]
            lemmatized_tokens = [lemmatize_text(x) for x in dialogue_list]
            for i in range(len(lemmatized_tokens)):
                tagger = build_tagger(original_tokens, lemmatized_tokens, topic_list[i], i)
                topic_tagger.extend(tagger)
            data_dict['dialogue'] = topic_tagger
            if args.postive_gen:
                for i in range(len(lemmatized_tokens)):
                    tagger = build_tagger(original_tokens, lemmatized_tokens, positive_topic_list[i], i)
                    positive_topic_tagger.extend(tagger)
                data_dict['positive_dialogue'] = positive_topic_tagger        
            if args.negative_gen:
                for i in range(len(lemmatized_tokens)):
                    negative_tagger_list = []
                    for j in range(len(negative_topic_list)):
                        tagger = build_tagger(original_tokens, lemmatized_tokens, negative_topic_list[i][j], i)
                        negative_tagger_list.extend(tagger)
                negative_topic_tagger.append(negative_tagger_list)
                data_dict['negative_dialogue'] = negative_topic_tagger
        else:
            # data_dict['dialogue'] = dialogue_list
            data_dict['dialogue'] = []
            if args.postive_gen:
                # data_dict['positive_dialogue'] = dialogue_list
                data_dict['positive_dialogue'] = []
            if args.negative_gen:
                # data_dict['negative_dialogue'] = dialogue_list
                data_dict['negative_dialogue'] = []

    data_dict = Dataset.from_dict(data_dict)

    return data_dict


def raw_data_loader(args):

    if 'dialogsum' in args.train_file:
        train_dict = load_from_dialogsum(args, args.train_file, 'train')
        val_dict = load_from_dialogsum(args, args.validation_file, 'val')
        test_dict = load_from_dialogsum(args, args.test_file, 'test')

    train_dict = utils.len_adjust(args, train_dict, 'train')
    val_dict = utils.len_adjust(args, val_dict, 'val')
    test_dict = utils.len_adjust(args, test_dict, 'test')

    raw_datasets = datasets.DatasetDict(
        {"train": train_dict, "validation": val_dict, "test": test_dict})

    return raw_datasets


@dataclass
class CustomWithNegativeDataCollator:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        if "positive_labels" in features[0].keys():
            positive_labels = [feature['positive_labels'] for feature in features]  
        else: 
            None
        if "negative_labels" in features[0].keys():
            negative_labels = [feature['negative_labels'] for feature in features]
        else: 
            None
        if "negative_labels_0" in features[0].keys():
            negative_labels_dict = {}
            negative_labels = [feature['negative_labels'] for feature in features]
            
        else: 
            None

        positive_inputs = [np.array(feature['positive_inputs']) for feature in features] if "positive_inputs" in features[0].keys() else None

        
        
        negative_input_0 = [np.array(feature['negative_input_ids_0']) for feature in features]
        # negative_input_1 = [np.array(feature['negative_input_ids_1']) for feature in features]
        # negative_input_2 = [np.array(feature['negative_input_ids_2']) for feature in features]
        
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)
        
        new_labels = [feature["labels"] for feature in features]
        
        # print(len(features))
        stack_features = self.tokenizer.pad(
            {"input_ids": input+positive_input+negative_input_0},
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        
        stack_features["labels"] = self.tokenizer.pad(
            {"input_ids": new_labels+new_labels+new_labels},
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )["input_ids"]
        
        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=stack_features["labels"])
            stack_features["decoder_input_ids"] = decoder_input_ids

        return stack_features

@dataclass
class CustomDataCollator:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        input = [np.array(feature['input_ids']) for feature in features]
        
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)
        
        new_labels = [feature["labels"] for feature in features]
        
        # print(len(features))
        stack_features = self.tokenizer.pad(
            {"input_ids": positive_input},
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        
        stack_features["labels"] = self.tokenizer.pad(
            {"input_ids": new_labels},
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )["input_ids"]
        
        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=stack_features["labels"])
            stack_features["decoder_input_ids"] = decoder_input_ids

        return stack_features

def data_processor(logger, args, accelerator, raw_datasets, tokenizer, model):
    ''' prepare dataset format for train/val/test '''
    def preprocess_function(examples):

        # summary - target
        if args.contrastive_loss:
            targets = examples[summary_column]
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)
            if args.topic_prompt_output or args.length_prompt_output:
                if args.postive_gen:
                    targets_postive = examples['positive_prompt']
                    with tokenizer.as_target_tokenizer():
                        labels_postive = tokenizer(targets_postive, max_length=max_target_length, padding=padding, truncation=True)
                if args.negative_gen:
                    targets_negative = examples['negative_prompt']
                    if args.negative_sample == 1:
                        with tokenizer.as_target_tokenizer():
                            labels_negative = tokenizer(targets_negative, max_length=max_target_length, padding=padding, truncation=True)
                    else:
                        labels_negative_list = []
                        for num in range(args.negative_sample):
                            key_summary_name = 'negative_summary_' + str(num)
                            targets_negative = examples[key_summary_name]
                            with tokenizer.as_target_tokenizer():
                                labels_negative = tokenizer(targets_negative, max_length=max_target_length, padding=padding, truncation=True)
                            labels_negative_list.append(labels_negative)

        else:
            targets = examples[summary_column]
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # dialogue - input
        inputs = examples[text_column]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)
        
        if args.contrastive_loss:
            if args.postive_gen:
                positive_inputs = examples['positive_prompt']
                positive_model_inputs = tokenizer(positive_inputs, max_length=args.max_source_length, padding=padding, truncation=True)
            if args.negative_gen:
                if args.negative_sample == 1:
                    negative_inputs = examples['negative_prompt']
                    negative_model_inputs = tokenizer(negative_inputs, max_length=args.max_source_length, padding=padding, truncation=True)
                else:
                    negative_model_inputs_list = []
                    for num in range(args.negative_sample):
                        key_prompt_name = 'negative_prompt_' + str(num)
                        negative_inputs = examples[key_prompt_name]
                        negative_model_inputs = tokenizer(negative_inputs, max_length=args.max_source_length, padding=padding, truncation=True)
                        negative_model_inputs_list.append(negative_model_inputs)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if args.contrastive_loss:
            if padding == "max_length" and args.ignore_pad_token_for_loss:
                labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]
                if args.topic_prompt_output or args.length_prompt_output:
                    if args.postive_gen:
                        labels_postive["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels_postive["input_ids"]]
                    if args.negative_gen:
                        if args.negative_sample == 1:
                            labels_negative["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels_negative["input_ids"]]
                        else:
                            for num in range(args.negative_sample):
                                labels_negative_list[num]["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels_negative_list[num]["input_ids"]]
        else:
            if padding == "max_length" and args.ignore_pad_token_for_loss:
                labels["input_ids"] = [
                    [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]

        if args.contrastive_loss:
            model_inputs["labels"] = labels["input_ids"]
            if args.postive_gen:
                model_inputs["positive_inputs"] = positive_model_inputs["input_ids"]
                if args.topic_prompt_output or args.length_prompt_output:
                    model_inputs["positive_labels"] = labels_postive["input_ids"]
            if args.negative_gen:
                if args.negative_sample == 1:
                    model_inputs["negative_inputs"] = negative_model_inputs["input_ids"]
                    if args.topic_prompt_output or args.length_prompt_output:
                        model_inputs["negative_labels"] = labels_negative["input_ids"]
                else:
                    for num in range(args.negative_sample):
                        key_model_inputs = 'negative_inputs_ids_' + str(num)
                        model_inputs[key_model_inputs] = negative_model_inputs_list[num]["input_ids"]
                        if args.topic_prompt_output or args.length_prompt_output:
                            key_model_inputs_labels = 'negative_labels_ids_' + str(num)
                            model_inputs[key_model_inputs_labels] = labels_negative_list[num]["input_ids"]
        else: 
            model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    # Preprocessing the datasets.
    column_names = raw_datasets["train"].column_names
    text_column = args.text_column
    summary_column = args.summary_column

    # Temporarily set max_target_length for training.
    max_target_length = args.max_target_length
    padding = "max_length" if args.pad_to_max_length else False

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            batch_size=1000,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset  = processed_datasets["validation"]
    test_dataset  = processed_datasets["test"]

    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id

    if args.contrastive_loss:
        data_collator = CustomWithNegativeDataCollator(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if accelerator.use_fp16 else None,
        )
        
        valid_data_collator = CustomDataCollator(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if accelerator.use_fp16 else None,
        )
        
        train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
        eval_dataloader = DataLoader(eval_dataset, collate_fn=valid_data_collator, batch_size=args.per_device_eval_batch_size)
        test_dataloader = DataLoader(test_dataset, collate_fn=valid_data_collator, batch_size=args.per_device_test_batch_size)
    else:
        data_collator = CustomDataCollator(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if accelerator.use_fp16 else None,
        )
    
        train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
        eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
        test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_test_batch_size)

    return (train_dataloader, eval_dataloader, test_dataloader), (train_dataset, eval_dataset, test_dataset)