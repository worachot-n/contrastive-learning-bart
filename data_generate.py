import json
import csv
import random
from random import randint
import warnings
import numpy as np
from collections.abc import Mapping
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import logging
import utils
from args import parse_args
from data_augment import data_augment


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

    data_dict = {'fname': id_list,
             'dialogue': dialogue_list,
             'summary': summary_list,
             'topic': topic_list}


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

    data_dict = {'fname': id_list,
             'dialogue': dialogue_list,
             'summary': summary_list,
             'topic': topic_list}
    
    if args.contrastive_loss:
        topic_set = set(topic_list)
        synonym_topic_list = []
        synonym_summary_list = []
        random_topic_list = []
        random_summary_list = []
        for topic, summary in zip(topic_list, summary_list):
            if args.synonym_replacement:
                synonym_topic, synonym_summary = data_augment(topic, summary, num_aug=1)
                synonym_topic_list.append(synonym_topic[0])
                synonym_summary_list.append(synonym_summary[0])
            if args.random_topic:
                random_topic = topic_set.difference(set([topic]))
                random_topic = random.choice(list(random_topic))
                while True: 
                    index_topic = random.choice(range(len(topic_list)))
                    random_topic = topic_list[index_topic]
                    if random_topic != topic: 
                        random_summary = summary_list[index_topic]
                        break
                random_topic_list.append(random_topic)
                random_summary_list.append(random_summary)
        if args.synonym_replacement:
            data_dict['synonym_topic'] = synonym_topic_list
            data_dict['synonym_summary'] = synonym_summary_list
        if args.random_topic:   
            data_dict['random_topic'] = random_topic_list
            data_dict['random_summary'] = random_summary_list
            
    data_dict_list = []
    for ids, dialogue, summary, topic, synonym_summary, synonym_topic, random_summary, random_topic in zip(id_list, dialogue_list, summary_list, topic_list, synonym_summary_list, 
                                                                                                            synonym_topic_list, random_summary_list, random_topic_list):

        new_dict = {}
        new_dict['fname'] = ids
        new_dict['dialogue'] = dialogue
        new_dict['summary'] = summary
        new_dict['topic'] = topic
        new_dict['synonym_summary'] = synonym_summary
        new_dict['synonym_topic'] = synonym_topic
        new_dict['random_summary'] = random_summary
        new_dict['random_topic'] = random_topic
        data_dict_list.append(new_dict)
        
    with open(f'./data/dialogsum_aug/dialogsum.{split_type}.jsonl', 'w') as outfile:
        for entry in data_dict_list:
            json.dump(entry, outfile)
            outfile.write('\n')

    return data_dict_list


# =  =  =  =  =  =  =  =  =  = Logging Setup =  =  =  =  =  =  =  =  =  =  =  = 

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


# = = = = = = = = = = = = = Main Process = = = = = = = = = = = = = = = = = =
def main():
    
    args = parse_args()
    # display parameters
    logging.info("*** Parameters ***")
    for item, value in vars(args).items():
        logging.info("{}: {}".format(item, value))
    logging.info("")
    
    train_dict = load_from_dialogsum(args, args.train_file, 'train')
    val_dict = load_from_dialogsum(args, args.validation_file, 'dev')
    test_dict = load_from_dialogsum(args, args.test_file, 'test')


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# main process
if __name__ == "__main__":
    main()
