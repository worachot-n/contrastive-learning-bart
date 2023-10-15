import json
import math

from tqdm import tqdm
from collections import Counter

import nltk
from nltk.util import ngrams
from nltk import word_tokenize, sent_tokenize

from datasets import Dataset


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    '''
        loss with label smoothing
        from fairseq, edit by Bin
    '''

    lprobs = lprobs[~target.eq(-100)]
    target = target[~target.eq(-100)]

    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)

    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)

    # mean()? Scared to break other math.
    # bin: change from sum to mean
    nll_loss = nll_loss.mean()
    smooth_loss = smooth_loss.mean()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def postprocess_text(preds, labels):
    '''
        use for decoding
    '''
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def len_adjust(args, split_dict, split_type=None):
    ''' add length to the input '''

    id_list = split_dict['id']
    dialogue_list = split_dict['dialogue']
    summary_list = split_dict['summary']
    topic_list = split_dict['topic']

    if args.len_input == 'no':
        new_dialogue_list = dialogue_list

    # elif args.len_input == 'surface':
    #     new_dialogue_list = []
    #     for dialogue in dialogue_list:
    #         utt_diag = dialogue.split('\n')
    #         num_utt = len(utt_diag)
    #         word_diag = [utt.split(' ') for utt in utt_diag]
    #         word_diag = [word for utt in word_diag for word in utt]
    #         num_word = len(word_diag)

    #         new_dialogue = 'Number of utterrance: {}. Length of Dialogue: {}. Dialogue: {}'.format(
    #             num_utt, num_word, dialogue)
    #         new_dialogue_list.append(new_dialogue)

    elif args.len_input == 'length':
        new_dialogue_list = []
        for dialogue, summary in zip(dialogue_list, summary_list):
            sum_len = len(summary.split(' '))
            new_dialogue = 'Length of Summary: {}. Dialogue: '.format(
                sum_len) + dialogue
            new_dialogue_list.append(new_dialogue)

    elif args.len_input == 'topic':
        new_dialogue_list = []
        for dialogue, summary, topic in zip(dialogue_list, summary_list, topic_list):
            topic_keyword = topic
            new_dialogue = 'Topic of Summary: {}. Dialogue: '.format(
                topic_keyword) + dialogue
            new_dialogue_list.append(new_dialogue)

    # elif args.len_input == 'topic-length':
    #     new_dialogue_list = []
    #     for dialogue, summary, topic in zip(dialogue_list, summary_list, topic_list):
    #         topic_keyword = topic
    #         sum_len = len(summary.split(' '))
    #         new_dialogue = 'Topic of Summary: {}. Length of Summary: {}. Dialogue: '.format(
    #             topic_keyword, sum_len) + dialogue
    #         new_dialogue_list.append(new_dialogue)
    
    elif args.len_input == 'topic-length':
        new_dialogue_list = []
        for dialogue, summary, topic in zip(dialogue_list, summary_list, topic_list):
            topic_keyword = topic
            sum_len = len(summary.split(' '))
            new_dialogue = 'Length of Summary: {}. Dialogue: '.format(
                topic_keyword, sum_len) + dialogue
            new_dialogue_list.append(new_dialogue)

    elif args.len_input == 'length-topic':
        new_dialogue_list = []
        for dialogue, summary, topic in zip(dialogue_list, summary_list, topic_list):
            topic_keyword = topic
            sum_len = len(summary.split(' '))
            new_dialogue = 'Length of Summary: {}. Topic of Summary: {}. Dialogue: '.format(
                sum_len, topic_keyword) + dialogue
            new_dialogue_list.append(new_dialogue)

    elif args.len_input == 'simple':
        new_dialogue_list = []
        for dialogue, summary in zip(dialogue_list, summary_list):
            sum_len = len(summary.split(' '))
            new_dialogue = 'Summary Length: {}. Dialogue: '.format(
                sum_len) + dialogue
            new_dialogue_list.append(new_dialogue)

    elif args.len_input == 'simple-topic':
        new_dialogue_list = []
        for dialogue, summary, topic in zip(dialogue_list, summary_list, topic_list):
            topic_keyword = topic
            sum_len = len(summary.split(' '))
            new_dialogue = 'Summary Length: {}. {}. Dialogue: '.format(
                sum_len, topic_keyword) + dialogue
            new_dialogue_list.append(new_dialogue)

    elif args.len_input == 'topic-simple':
        new_dialogue_list = []
        for dialogue, summary, topic in zip(dialogue_list, summary_list, topic_list):
            topic_keyword = topic
            sum_len = len(summary.split(' '))
            new_dialogue = '{}. Summary Length: {}. Dialogue: '.format(
                topic_keyword, sum_len) + dialogue
            new_dialogue_list.append(new_dialogue)

    elif args.len_input == 'topic-word':
        new_dialogue_list = []
        for dialogue, summary, topic in zip(dialogue_list, summary_list, topic_list):
            topic_keyword = topic
            new_dialogue = '{}. Dialogue: '.format(
                topic_keyword) + dialogue
            new_dialogue_list.append(new_dialogue)

    elif args.len_input == 'topic-last-length':
        new_dialogue_list = []
        for dialogue, summary, topic in zip(dialogue_list, summary_list, topic_list):
            topic_keyword = topic
            sum_len = len(summary.split(' '))
            new_dialogue = 'Topic of Summary: {}. Dialogue: {} Length of Summary: {}'.format(
                topic_keyword, dialogue, sum_len)
            new_dialogue_list.append(new_dialogue)

    if args.len_output == 'no' or split_type == 'val' or split_type == 'test':
        new_summary_list = summary_list

    elif args.len_output == 'length':
        new_summary_list = []
        for summary in summary_list:
            sum_len = len(summary.split(' '))
            new_summary = 'Length of Summary: {}. Summary: '.format(
                sum_len) + summary
            new_summary_list.append(new_summary)

    elif args.len_output == 'topic':
        new_summary_list = []
        for summary, topic in zip(summary_list, topic_list):
            topic_keyword = topic
            new_summary = 'Topic of Summary: {}. Summary: '.format(
                topic_keyword) + summary
            new_summary_list.append(new_summary)

    elif args.len_output == 'topic-length':
        new_summary_list = []
        for summary, topic in zip(summary_list, topic_list):
            topic_keyword = topic
            sum_len = len(summary.split(' '))
            new_summary = 'Topic of Summary: {}. Length of Summary: {}. Summary: '.format(
                topic_keyword, sum_len) + summary
            new_summary_list.append(new_summary)

    elif args.len_output == 'length-topic':
        new_summary_list = []
        for summary, topic in zip(summary_list, topic_list):
            topic_keyword = topic
            sum_len = len(summary.split(' '))
            new_summary = 'Length of Summary: {}. Topic of Summary: {}. Summary: '.format(
                sum_len, topic_keyword) + summary
            new_summary_list.append(new_summary)
    
    split_dict = {
        'id': id_list,
        'dialogue': new_dialogue_list,
        'summary': new_summary_list,
        'topic': topic_list
    }

    split_dict = Dataset.from_dict(split_dict)

    return split_dict