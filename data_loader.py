import json
import csv
import random
from random import randint
import warnings
import numpy as np
from collections.abc import Mapping
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import datasets
from datasets import Dataset
import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy

import utils
from special_token import simple_tokenize, lemmatize_text, build_tagger


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
        if args.synonym_replacement:
            # topic
            synonym_topic_list = [sample['synonym_topic'] for sample in data]
            # summary
            synonym_summary_list = [sample['synonym_summary'] for sample in data]
            data_dict['synonym_topic'] = synonym_topic_list
            data_dict['synonym_summary'] = synonym_summary_list
        if args.random_topic:
            # topic
            random_topic_list = [sample['random_topic'] for sample in data]
            # summary
            random_summary_list = [sample['random_summary'] for sample in data]
            data_dict['random_topic'] = random_topic_list
            data_dict['random_summary'] = random_summary_list

    if args.tagging == "topic" or args.tagging == "prompt":
        original_tagger = []
        original_tokens = [simple_tokenize(x) for x in dialogue_list]
        lemmatized_tokens = [lemmatize_text(x) for x in dialogue_list]
        for i in range(len(lemmatized_tokens)):
            tagger = build_tagger(original_tokens, lemmatized_tokens, topic_list[i], i)
            original_tagger.extend(tagger)
        data_dict['dialogue'] = original_tagger
        if args.contrastive_loss:
            if args.synonym_replacement:
                synonym_tagger = []
                for i in range(len(lemmatized_tokens)):
                    tagger = build_tagger(original_tokens, lemmatized_tokens, synonym_topic_list[i], i)
                    synonym_tagger.extend(tagger)
                data_dict['synonym_dialogue'] = synonym_tagger
            if args.random_topic:
                random_tagger = []
                for i in range(len(lemmatized_tokens)):
                    tagger = build_tagger(original_tokens, lemmatized_tokens, random_topic_list[i], i)
                    random_tagger.extend(tagger)
                data_dict['random_dialogue'] = random_tagger

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
        inputs = [np.array(feature['input_ids']) for feature in features] if "input_ids" in features[0].keys() else None
        if "synonym_labels" in features[0].keys():
            synonym_labels = [feature['synonym_labels'] for feature in features]  
        else: 
            None
        if "random_labels" in features[0].keys():
            random_labels = [feature['random_labels'] for feature in features]
        else: 
            None
        if "synonym_inputs" in features[0].keys():
            synonym_inputs = [np.array(feature['synonym_inputs']) for feature in features]
        else: 
            None
        if "random_inputs" in features[0].keys():
            random_inputs = [np.array(feature['random_inputs']) for feature in features]
        else: 
            None

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
                if "synonym_labels" in features[0].keys():
                    remainder_synonym = [self.label_pad_token_id] * (max_label_length - len(feature["synonym_labels"]))
                if "random_labels" in features[0].keys():
                    remainder_random = [self.label_pad_token_id] * (max_label_length - len(feature["random_labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                    if "synonym_labels" in features[0].keys():
                        feature["synonym_labels"] = (
                            feature["synonym_labels"] + remainder_synonym if padding_side == "right" else remainder_synonym + feature["synonym_labels"]
                        )
                    if "random_labels" in features[0].keys():
                        feature["random_labels"] = (
                            feature["random_labels"] + remainder_random if padding_side == "right" else remainder_random + feature["random_labels"]
                        )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                    if "synonym_labels" in features[0].keys():
                        feature["synonym_labels"] = np.concatenate([feature["synonym_labels"], remainder_synonym]).astype(np.int64)
                    if "random_labels" in features[0].keys():
                        feature["random_labels"] = np.concatenate([feature["random_labels"], remainder_random]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)
                    if "synonym_labels" in features[0].keys():
                        feature["synonym_labels"] = np.concatenate([remainder_synonym, feature["synonym_labels"]]).astype(np.int64)
                    if "random_labels" in features[0].keys():
                        feature["random_labels"] = np.concatenate([remainder_random, feature["random_labels"]]).astype(np.int64)
        
        new_labels = [feature["labels"] for feature in features]
        if "synonym_labels" in features[0].keys():
            new_synonym_labels = [feature["synonym_labels"] for feature in features]
        if "random_labels" in features[0].keys():
            new_random_labels = [feature["random_labels"] for feature in features]

        if "synonym_inputs" in features[0].keys() and "random_inputs" not in features[0].keys():
            stack_features = self.tokenizer.pad(
                {"input_ids": inputs+synonym_inputs},
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=return_tensors,
            )
        elif "synonym_inputs" not in features[0].keys() and "random_inputs" in features[0].keys():
            stack_features = self.tokenizer.pad(
                {"input_ids": inputs+random_inputs},
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=return_tensors,
            )
        elif "synonym_inputs" in features[0].keys() and "random_inputs" in features[0].keys():
            stack_features = self.tokenizer.pad(
                {"input_ids": inputs+synonym_inputs+random_inputs},
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=return_tensors,
            )
        else:
            stack_features = self.tokenizer.pad(
                {"input_ids": inputs},
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=return_tensors,
            )

        if "synonym_labels" in features[0].keys() and "random_labels" not in features[0].keys():
            stack_features["labels"] = self.tokenizer.pad(
                {"input_ids": new_labels+synonym_labels},
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=return_tensors,
            )["input_ids"]
        elif "synonym_labels" not in features[0].keys() and "random_labels" in features[0].keys():
            stack_features["labels"] = self.tokenizer.pad(
                {"input_ids": new_labels+random_labels},
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=return_tensors,
            )["input_ids"]
        elif "synonym_labels" in features[0].keys() and "random_labels" in features[0].keys():
            stack_features["labels"] = self.tokenizer.pad(
                {"input_ids": new_labels+synonym_labels+random_labels},
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=return_tensors,
            )["input_ids"]
        else:
            if "synonym_inputs" in features[0].keys() and "random_inputs" not in features[0].keys():
                stack_features["labels"] = self.tokenizer.pad(
                    {"input_ids": new_labels+new_labels},
                    padding=self.padding,
                    max_length=self.max_length,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors=return_tensors,
                )["input_ids"]
            if "synonym_inputs" not in features[0].keys() and "random_inputs" in features[0].keys():
                stack_features["labels"] = self.tokenizer.pad(
                    {"input_ids": new_labels+new_labels},
                    padding=self.padding,
                    max_length=self.max_length,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors=return_tensors,
                )["input_ids"]
            if "synonym_inputs" in features[0].keys() and "random_inputs" in features[0].keys():
                stack_features["labels"] = self.tokenizer.pad(
                    {"input_ids": new_labels+new_labels+new_labels},
                    padding=self.padding,
                    max_length=self.max_length,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors=return_tensors,
                )["input_ids"]
            else:
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
        inputs = [np.array(feature['input_ids']) for feature in features]
        
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

        stack_features = self.tokenizer.pad(
            {"input_ids": inputs},
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
            if args.contrastive_decoder:
                if args.synonym_replacement:
                    targets_synonym = examples['synonym_summary']
                    with tokenizer.as_target_tokenizer():
                        labels_synonym = tokenizer(targets_synonym, max_length=max_target_length, padding=padding, truncation=True)
                if args.random_topic:
                    targets_random = examples['random_summary']
                    with tokenizer.as_target_tokenizer():
                        labels_random = tokenizer(targets_random, max_length=max_target_length, padding=padding, truncation=True)

        else:
            targets = examples[summary_column]
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # dialogue - input
        inputs = examples[text_column]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)
        
        if args.contrastive_loss:
            if args.synonym_replacement:
                synonym_inputs = examples['synonym_prompt']
                synonym_model_inputs = tokenizer(synonym_inputs, max_length=args.max_source_length, padding=padding, truncation=True)
            if args.random_topic:
                random_inputs = examples['random_prompt']
                random_model_inputs = tokenizer(random_inputs, max_length=args.max_source_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if args.contrastive_loss:
            if padding == "max_length" and args.ignore_pad_token_for_loss:
                labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]
                if args.contrastive_decoder:
                    if args.synonym_replacement:
                        labels_synonym["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels_synonym["input_ids"]]
                    if args.random_topic:
                        labels_random["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels_random["input_ids"]]
        else:
            if padding == "max_length" and args.ignore_pad_token_for_loss:
                labels["input_ids"] = [
                    [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]

        if args.contrastive_loss:
            model_inputs["labels"] = labels["input_ids"]
            if args.synonym_replacement:
                model_inputs["synonym_inputs"] = synonym_model_inputs["input_ids"]
                if args.contrastive_decoder:
                    model_inputs["synonym_labels"] = labels_synonym["input_ids"]
            if args.random_topic:
                model_inputs["random_inputs"] = random_model_inputs["input_ids"]
                if args.contrastive_decoder:
                    model_inputs["random_labels"] = labels_random["input_ids"]
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