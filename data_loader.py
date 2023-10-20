import json
import csv
import random

import datasets
from datasets import Dataset
from torch.utils.data import DataLoader

from transformers import DataCollatorForSeq2Seq

import utils

from topic_tagger import simple_tokenize, lemmatize_text, build_tagger

import numpy as np


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

    negative_topic_list = []
    for topic in topic_list:
        negative_topic = random.choice(topic_list)
        if negative_topic == topic:
            negative_topic = random.choice(negative_topic)
        negative_topic_list.append(negative_topic)
        

    if args.topic_tagger:
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
    else:
        data_dict = {'id': id_list,
                     'dialogue': dialogue_list,
                     'summary': summary_list,
                     'topic': topic_list,
                     'negative_topic': negative_topic_list}

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

class CustomDataCollator:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def __call__(self, examples):
        # positive_input_ids = examples['dialogue']
        positive_input = [np.array(example['dialogue']) for example in examples]
        # positive_input_ids = [example['dialogue'][example['dialogue'] != 1] for example in examples]
        # negative_input_ids = examples['negative_dialogue']
        negative_input = [np.array(example['negative_dialogue']) for example in examples]
        # negative_input_ids = [example['negative_dialogue'][example['dialogue'] != 1] for example in examples]
        # summary_input_ids = examples['summary']
        summary_input = [np.array(example['summary']) for example in examples]
        # summary_input_ids = [example['summary'][example['dialogue'] != 1] for example in examples]

        positive_input_ids = [example[example != 1] for example in positive_input]
        negative_input_ids = [example[example != 1] for example in negative_input]
        summary_input_ids = [example[example != 1] for example in summary_input]
        # summary_input_ids = [example['summary'] for example in examples]
            
        # inputs["input_ids"] = tokenizer.pad
        # negative_inputs["input_ids"] = tokenizer.pad
        
        batch = self.tokenizer.pad(encoded_inputs={"input_ids": positive_input_ids+ negative_input_ids}, padding=True, return_tensors='pt')
        # batch["decoder_input_ids"] = torch.stack((inputs["labels"], inputs["labels"]))
        # batch["decoder_attention_mask"] = torch.stack((inputs["labels"], inputs["decoder_attention_mask"]))
        batch["decoder_input_ids"] = self.tokenizer.pad(encoded_inputs={"input_ids": summary_input_ids+summary_input_ids}, padding=True, return_tensors='pt')["input_ids"]
        # summary = [[(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in summary_input_ids]
        summary = self.tokenizer.pad(encoded_inputs={"input_ids": summary_input_ids+summary_input_ids}, padding=True, return_tensors='pt')["input_ids"]
        summary[summary == 1] = -100
        batch["labels"] = summary
        # summary =  self.tokenizer.pad(encoded_inputs={"input_ids": summary_input_ids+summary_input_ids}, padding=True, return_tensors='pt')["input_ids"]
        # summary = [[(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in summary_input_ids]

        return batch

class CustomDataCollatorValidate:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def __call__(self, examples):
        # positive_input_ids = examples['dialogue']
        positive_input = [np.array(example['dialogue']) for example in examples]
        # positive_input_ids = [example['dialogue'][example['dialogue'] != 1] for example in examples]
        # negative_input_ids = examples['negative_dialogue']
        # negative_input = [np.array(example['negative_dialogue']) for example in examples]
        # negative_input_ids = [example['negative_dialogue'][example['dialogue'] != 1] for example in examples]
        # summary_input_ids = examples['summary']
        summary_input = [np.array(example['summary']) for example in examples]
        # summary_input_ids = [example['summary'][example['dialogue'] != 1] for example in examples]

        positive_input_ids = [example[example != 1] for example in positive_input]
        # negative_input_ids = [example[example != 1] for example in negative_input]
        summary_input_ids = [example[example != 1] for example in summary_input]
        # summary_input_ids = [example['summary'] for example in examples]
            
        # inputs["input_ids"] = tokenizer.pad
        # negative_inputs["input_ids"] = tokenizer.pad
        
        batch = self.tokenizer.pad(encoded_inputs={"input_ids": positive_input_ids}, padding=True, return_tensors='pt')
        # batch["decoder_input_ids"] = torch.stack((inputs["labels"], inputs["labels"]))
        # batch["decoder_attention_mask"] = torch.stack((inputs["labels"], inputs["decoder_attention_mask"]))
        batch["decoder_input_ids"] = self.tokenizer.pad(encoded_inputs={"input_ids": summary_input_ids}, padding=True, return_tensors='pt')["input_ids"]
        # summary = [[(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in summary_input_ids]
        summary = self.tokenizer.pad(encoded_inputs={"input_ids": summary_input_ids}, padding=True, return_tensors='pt')["input_ids"]
        summary[summary == 1] = -100
        batch["labels"] = summary
        # summary =  self.tokenizer.pad(encoded_inputs={"input_ids": summary_input_ids+summary_input_ids}, padding=True, return_tensors='pt')["input_ids"]
        # summary = [[(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in summary_input_ids]

        return batch

def data_processor(logger, args, accelerator, raw_datasets, tokenizer, model):
    ''' prepare dataset format for train/val/test '''
    def preprocess_function(examples):
        positive_documents = examples['dialogue']
        negative_documents = examples['negative_dialogue']
        source_summaries = examples['summary']

        # Tokenize and create input tensors
        inputs = tokenizer(
            positive_documents,
            # negative_summaries,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_source_length  # Adjust as needed
        )
        
        # Tokenize and create input tensors
        negative_inputs = tokenizer(
            negative_documents,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_source_length  # Adjust as needed
        )
        
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                source_summaries,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_target_length  # Adjust as needed
            )
        
        # batch = tokenizer.pad(encoded_inputs={"input_ids": inputs["input_ids"].squeeze().tolist() + negative_inputs["input_ids"].squeeze().tolist()}, padding=True, return_tensors='pt')
        # # batch["decoder_input_ids"] = torch.stack((inputs["labels"], inputs["labels"]))
        # # batch["decoder_attention_mask"] = torch.stack((inputs["labels"], inputs["decoder_attention_mask"]))
        # batch["decoder_input_ids"] = tokenizer.pad(encoded_inputs={"input_ids": labels["input_ids"].squeeze().tolist()+labels["input_ids"].squeeze().tolist()}, padding=False, return_tensors='pt')["input_ids"]
        # labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]
        # batch["labels"] =  tokenizer.pad(encoded_inputs={"input_ids": labels["input_ids"]+labels["input_ids"]}, padding=True, return_tensors='pt')["input_ids"]

        # return batch
        model_inputs = inputs
        model_inputs["dialogue"] = inputs["input_ids"]
        model_inputs["negative_dialogue"] = negative_inputs["input_ids"]
        model_inputs["summary"] = labels["input_ids"]

        return model_inputs

    prefix = args.source_prefix if args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names

    # Get the column names for input/target.
    text_column = args.text_column
    if text_column not in column_names:
        raise ValueError(
            f"--text_column' value '{args.text_column}' needs to be one of: {', '.join(column_names)}"
        )

    summary_column = args.summary_column
    if summary_column not in column_names:
        raise ValueError(
            f"--summary_column' value '{args.summary_column}' needs to be one of: {', '.join(column_names)}"
        )

    # Temporarily set max_target_length for training.
    max_target_length = args.max_target_length
    padding = "max_length" if args.pad_to_max_length else False

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            batch_size=100,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset  = processed_datasets["validation"]
    test_dataset  = processed_datasets["test"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    # data_collator = DataCollatorForSeq2Seq(
    #     tokenizer,
    #     model=model,
    #     label_pad_token_id=label_pad_token_id,
    #     pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    # )
    data_collator = CustomDataCollator(
        tokenizer,
        model=model,
    )
    
    validate_data_collator = CustomDataCollatorValidate(
        tokenizer,
        model=model,
    )

    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=validate_data_collator, batch_size=args.per_device_eval_batch_size)
    test_dataloader = DataLoader(test_dataset, collate_fn=validate_data_collator, batch_size=args.per_device_test_batch_size)

    return (train_dataloader, eval_dataloader, test_dataloader), (train_dataset, eval_dataset, test_dataset)
