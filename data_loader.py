import json
import csv
import random

import datasets
from datasets import Dataset
from torch.utils.data import DataLoader

from transformers import DataCollatorForSeq2Seq

import utils

from topic_tagger import simple_tokenize, lemmatize_text, build_tagger

import random
import warnings
from collections.abc import Mapping
from random import randint
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import numpy as np

from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch


def load_from_dialogsum(args, file_path):
    ''' load dialoguesum jsonl data '''

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

        id_list1 = [id+"_sum1" for id in id_list]
        id_list2 = [id+"_sum2" for id in id_list]
        id_list3 = [id+"_sum3" for id in id_list]

        id_list = id_list1 + id_list2 + id_list3
        dialogue_list = dialogue_list + dialogue_list + dialogue_list

        # summary
        summary_list1 = [sample['summary1'] for sample in data]
        summary_list2 = [sample['summary2'] for sample in data]
        summary_list3 = [sample['summary3'] for sample in data]

        summary_list = summary_list1 + summary_list2 + summary_list3

        # topic
        topic_list1 = [sample['topic1'] for sample in data]
        topic_list2 = [sample['topic2'] for sample in data]
        topic_list3 = [sample['topic3'] for sample in data]

        topic_list = topic_list1 + topic_list2 + topic_list3

    if args.contrastive_loss:
        negative_topic_list = []
        for topic in topic_list:
            negative_topic = random.choice(topic_list)
            if negative_topic == topic:
                negative_topic = random.choice(negative_topic)
            negative_topic_list.append(negative_topic)
        

    if not args.contrastive_loss and args.topic_tagger:
        topic_tagger = []
        original_tokens = [simple_tokenize(x) for x in dialogue_list]
        lemmatized_tokens = [lemmatize_text(x) for x in dialogue_list]
        for i in range(len(lemmatized_tokens)):
            tagger = build_tagger(original_tokens, lemmatized_tokens, topic_list[i], i)
            topic_tagger.extend(tagger)
        data_dict = {'id': id_list,
                     'dialogue': topic_tagger,
                     'summary': summary_list,
                     'topic': topic_list}
        
    elif args.contrastive_loss and args.topic_tagger:
        topic_tagger = []
        negative_topic_tagger = []
        original_tokens = [simple_tokenize(x) for x in dialogue_list]
        lemmatized_tokens = [lemmatize_text(x) for x in dialogue_list]
        for i in range(len(lemmatized_tokens)):
            tagger = build_tagger(original_tokens, lemmatized_tokens, topic_list[i], i)
            topic_tagger.extend(tagger)
        for i in range(len(lemmatized_tokens)):
            tagger = build_tagger(original_tokens, lemmatized_tokens, negative_topic_list[i], i)
            negative_topic_tagger.extend(tagger)
        data_dict = {'id': id_list,
                     'dialogue': topic_tagger,
                     'negative_dialogue': negative_topic_tagger,
                     'summary': summary_list,
                     'topic': topic_list,
                     'negative_topic': negative_topic_list}

    elif args.contrastive_loss and not args.topic_tagger:
        data_dict = {'id': id_list,
                     'dialogue': dialogue_list,
                     'summary': summary_list,
                     'topic': topic_list,
                     'negative_topic': negative_topic_list}

    else: 
        data_dict = {'id': id_list,
                     'dialogue': dialogue_list,
                     'summary': summary_list,
                     'topic': topic_list}
    data_dict = Dataset.from_dict(data_dict)

    return data_dict


def raw_data_loader(args):
    ''' load raw datasets from csv files '''

    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
    if args.test_file is not None:
        data_files["test"] = args.test_file

    if 'dialogsum' in args.train_file:
        train_dict = load_from_dialogsum(args, args.train_file)
        val_dict = load_from_dialogsum(args, args.validation_file)
        test_dict = load_from_dialogsum(args, args.test_file)

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
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        positive_input = [np.array(feature['input_ids']) for feature in features]
        negative_input = [np.array(feature['negative_input_ids']) for feature in features]
        
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
            {"input_ids": positive_input+negative_input},
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        
        stack_features["labels"] = self.tokenizer.pad(
            {"input_ids": new_labels+new_labels},
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
class CustomWithNegativeDeocoderDataCollator:
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
        negative_labels = [feature["negative_labels"] for feature in features] if "negative_labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        positive_input = [np.array(feature['input_ids']) for feature in features]
        negative_input = [np.array(feature['negative_input_ids']) for feature in features]

        
        if labels is not None and negative_labels is not None:
            max_label_length = max(len(l) for l in labels)
            max_negative_label_length = max(len(l) for l in negative_labels)
            if self.pad_to_multiple_of is not None:
                max_negative_label_length = (
                    (max_negative_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )
                max_negative_label_length = (
                    (max_negative_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                negative_remainder = [self.label_pad_token_id] * (max_negative_label_length - len(feature["negative_labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                    feature["negative_labels"] = (
                        feature["negative_labels"] + negative_remainder if padding_side == "right" else negative_remainder + feature["negative_labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                    feature["negative_labels"] = np.concatenate([feature["negative_labels"], negative_remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)
                    feature["negative_labels"] = np.concatenate([negative_remainder, feature["negative_labels"]]).astype(np.int64)
        else:
            print("Negative decoders are errors")
        
        new_labels = [feature["labels"] for feature in features]
        new_negative_labels = [feature["negative_labels"] for feature in features]
        
        # print(len(features))
        stack_features = self.tokenizer.pad(
            {"input_ids": positive_input+negative_input},
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        
        stack_features["labels"] = self.tokenizer.pad(
            {"input_ids": new_labels+new_negative_labels},
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
        positive_input = [np.array(feature['input_ids']) for feature in features]
        
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
            negative_decoders = examples['negative_summary']
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)
                negative_model_decoders = tokenizer(negative_decoders, max_length=max_target_length, padding=padding, truncation=True)
        else:
            targets = examples[summary_column]
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # dialogue - input
        inputs = examples[text_column]

        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)
        
        if args.contrastive_loss:
            negative_inputs = examples['negative_dialogue']
            negative_model_inputs = tokenizer(negative_inputs, max_length=args.max_source_length, padding=padding, truncation=True)


        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if args.contrastive_loss:
            if padding == "max_length" and args.ignore_pad_token_for_loss:
                labels["input_ids"] = [
                    [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]
                negative_model_decoders["input_ids"] = [
                    [(l if l != tokenizer.pad_token_id else -100) for l in negative_model_decoder] for negative_model_decoder in negative_model_decoders["input_ids"]
                ]
        else:
            if padding == "max_length" and args.ignore_pad_token_for_loss:
                labels["input_ids"] = [
                    [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]
        if args.contrastive_loss:
            model_inputs["negative_input_ids"] = negative_model_inputs["input_ids"]
            model_inputs["labels"] = labels["input_ids"]
            model_inputs["negative_labels"] = negative_model_decoders["input_ids"]
        else: 
            model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names

    # Get the column names for input/target.
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

    # Log a few random samples from the training set:
    # for index in random.sample(range(len(train_dataset)), 1):
    #     logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id

    if args.contrastive_loss:
        data_collator = CustomWithNegativeDataCollator(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if accelerator.use_fp16 else None,
        )

        decoder_data_collator = CustomWithNegativeDeocoderDataCollator(
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
        # if args.decoder_topic_tagger:
        #     train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=decoder_data_collator, batch_size=args.per_device_train_batch_size)
        # else:
        #     train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
        train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=decoder_data_collator, batch_size=args.per_device_train_batch_size)
        # train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
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