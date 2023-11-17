import json
import math

from tqdm import tqdm
from datasets import Dataset
from collections import Counter

import nltk
from nltk.util import ngrams
from nltk import word_tokenize, sent_tokenize


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
    id_list = split_dict['id']
    dialogue_list = split_dict['dialogue']
    summary_list = split_dict['summary']
    topic_list = split_dict['topic']
    if args.contrastive_loss:
        if args.synonym_replacement:
            synonym_topic_list = split_dict['synonym_topic']
            if args.tagging == "word" or args.tagging == "prompt":
                synonym_dialogue_list = split_dict['synonym_dialogue']
            else:
                synonym_dialogue_list = dialogue_list
        else:
            synonym_dialogue_list = dialogue_list
            synonym_topic_list = topic_list
            
        if args.random_topic:
            random_topic_list = split_dict['random_topic']
            if args.tagging == "word" or args.tagging == "prompt":
                random_dialogue_list = split_dict['random_dialogue']
            else:
                random_dialogue_list = dialogue_list
        else:
            random_dialogue_list = dialogue_list
            random_topic_list = topic_list
    else:
        synonym_dialogue_list = dialogue_list
        synonym_topic_list = topic_list
        random_dialogue_list = dialogue_list
        random_topic_list = topic_list
        
    new_prompt_list = []
    new_synonym_prompt_list = []
    new_random_prompt_list = []

    for dialogue, summary, topic, synonym_dialogue, random_dialogue, synonym_topic, random_topics in zip(dialogue_list, summary_list, topic_list,
                                                                                                             synonym_dialogue_list, random_dialogue_list,
                                                                                                             synonym_topic_list, random_topic_list):
        if args.topic_prompt_input or args.length_prompt_input:
            new_dialogue = f'Dialogue: {dialogue}'
        else:
            new_dialogue = dialogue
        new_synonym_dialogue = f'Dialogue: {synonym_dialogue}'
        new_random_dialogue = f'Dialogue: {random_dialogue}'

        if args.topic_prompt_input:
            if args.tagging == "word":
                new_topic_input = f'Topic of Summary: <topic>{topic}</topic>. '
                if args.synonym_replacement:
                    new_synonym_topic_input = f'Topic of Summary: <topic>{synonym_topic}</topic>. '
                if args.random_topic:
                    new_random_topic_input = f'Topic of Summary: <topic>{random_topics}</topic>. '
            elif args.tagging == "prompt":
                new_topic_input = f'<topic>Topic of Summary: {topic}</topic>. '
                if args.synonym_replacement:
                    new_synonym_topic_input = f'<topic>Topic of Summary: {synonym_topic}</topic>. '
                if args.random_topic:
                    new_random_topic_input = f'<topic>Topic of Summary: {random_topics}</topic>. '
            else:
                new_topic_input = f'Topic of Summary: {topic}. '
                if args.synonym_replacement:
                    new_synonym_topic_input = f'Topic of Summary: {synonym_topic}. '
                if args.random_topic:
                    new_random_topic_input = f'Topic of Summary: {random_topics}. '
        else:
            new_topic_input = ''
            if args.synonym_replacement:
                new_synonym_topic_input = ''
            if args.random_topic:
                new_random_topic_input = ''
        if args.length_prompt_input:
            sum_len = len(summary.split(' '))
            new_length_input = f'Length of Summary: {sum_len}. '
            if args.synonym_replacement:
                synonym_sum_len = len(summary.split(' '))
                new_synonym_length_input = f'Length of Summary: {synonym_sum_len}. '
            if args.random_topic:
                random_sum_len = len(summary.split(' '))
                new_random_length_input = f'Length of Summary: {random_sum_len}. '

        else:
            new_topic_input = ''
                
        new_prompt = new_topic_input + new_length_input + new_dialogue
        new_prompt_list.append(new_prompt)
        new_summary_list.append(summary)
        if args.synonym_replacement:
            new_synonym_prompt = new_synonym_topic_input + new_synonym_length_input + new_synonym_dialogue
            new_synonym_prompt_list.append(new_synonym_prompt)
        if args.random_topic:
            new_random_prompt = new_random_topic_input + new_random_length_input + new_random_dialogue
            new_random_prompt_list.append(new_random_prompt)                                                                                         

    split_dict = {
        'id': id_list,
        'prompt': new_prompt_list,
        'summary': new_summary_list,
        'topic': topic_list,
    }

    if args.synonym_replacement:
        split_dict['synonym_prompt'] = new_synonym_prompt_list
        split_dict['synonym_topic'] = synonym_topic_list

    if args.random_topic:
        split_dict['random_prompt'] = new_random_prompt_list
        split_dict['random_topic'] = random_topic_list

    split_dict = Dataset.from_dict(split_dict)

    return split_dict