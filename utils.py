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
        if args.postive_gen:
            positive_topic_list = split_dict['positive_topic']
            if args.topic_tagger:
                positive_dialogue_list = split_dict['positive_dialogue']
            else:
                positive_dialogue_list = dialogue_list
        else:
            positive_topic_list = topic_list
        if args.negative_gen:
            negative_topic_list = split_dict['negative_topic']
            if args.topic_tagger:
                negative_dialogue_list = split_dict['negative_dialogue']
            else:
                negative_dialogue_list = dialogue_list
                negative_dialogue_list = dialogue_list
        else:
            negative_topic_list = topic_list
            negative_topic_list = topic_list
    
    new_prompt_list = []
    new_positive_prompt_list = []
    new_negative_prompt_list = []
    new_summary_list = []
    new_positive_summary_list = []
    new_negative_summary_list = []
    for dialogue, summary, topic, positive_dialogue, negative_dialogue, positive_topic, negative_topics in zip(dialogue_list, summary_list, topic_list,
                                                                                                             positive_dialogue_list, negative_dialogue_list,
                                                                                                             positive_topic_list, negative_topic_list):
        new_dialogue = f'Dialogue: {dialogue}'
        new_positive_dialogue = f'Dialogue: {positive_dialogue}'
        new_negative_dialogue = f'Dialogue: {negative_dialogue}'
        new_summary = f'Summary: {dialogue}'
        if args.topic_prompt_input:
            if args.topic_tagger:
                new_topic_input = f'<tp>Topic of Summary</tp>: {topic}. '
                if args.postive_gen:
                    new_positive_topic_input = f'<tp>Topic of Summary</tp>: {positive_topic}. '
                if args.negative_gen:
                    new_negative_topic_input = f'<tp>Topic of Summary</tp>: {negative_topics}. '
            else:
                new_topic_input = f'Topic of Summary: {topic}. '
                if args.postive_gen:
                    new_positive_topic_input = f'Topic of Summary: {positive_topic}. '
                if args.negative_gen:
                    new_negative_topic_input = f'Topic of Summary: {negative_topics}. '
        else:
            new_topic_input = ''
            if args.postive_gen:
                new_positive_topic_input = ''
            if args.negative_gen:
                new_negative_topic_input = ''
        if args.length_prompt_input:
            sum_len = len(summary.split(' '))
            new_length_input = f'Length of Summary: {sum_len}. '
        else:
            new_length_input = ''
        new_prompt = new_topic_input + new_length_input + new_dialogue
        new_prompt_list.append(new_prompt)                                                                                           
        if args.postive_gen:
            new_positive_prompt = new_positive_topic_input + new_length_input + new_positive_dialogue
            new_positive_prompt_list.append(new_positive_prompt)
        if args.negative_gen:
            new_negative_prompt = new_negative_topic_input + new_length_input + new_negative_dialogue
            new_negative_prompt_list.append(new_negative_prompt)
        if split_type == 'train':
            if args.topic_prompt_output:
                if args.topic_tagger:
                    new_topic_output = f'<tp>Topic of Summary</tp>: {topic}. '
                    if args.postive_gen:
                        new_positive_topic_output = f'<tp>Topic of Summary</tp>: {positive_topic}. '
                    if args.negative_gen:
                        new_negative_topic_output = f'<tp>Topic of Summary</tp>: {negative_topics}. '
                else:
                    new_topic_output = f'Topic of Summary: {topic}. '
                    if args.postive_gen:
                        new_positive_topic_output = f'Topic of Summary: {positive_topic}. '
                    if args.negative_gen:
                        new_negative_topic_output = f'Topic of Summary: {negative_topics}. '
            else:
                new_topic_output = ''
                if args.postive_gen:
                    new_positive_topic_output = ''
                if args.negative_gen:
                    new_negative_topic_output = ''
            if args.length_prompt_output:
                sum_len = len(summary.split(' '))
                new_length_output = f'Length of Summary: {sum_len}. '
            else:
                new_length_output = ''
            new_summary_all = new_topic_output + new_length_output + new_summary
            new_summary_list.append(new_summary_all)   
            if args.postive_gen:
                new_positive_summary = new_positive_topic_output + new_length_output + new_summary
                new_positive_summary_list.append(new_positive_summary)
            if args.negative_gen:
                new_negative_summary = new_negative_topic_output + new_length_output + new_summary
                new_negative_summary_list.append(new_negative_summary)

        else:
            new_summary_all = new_summary
            new_summary_list.append(new_summary_all)   
            if args.postive_gen:
                new_positive_summary = ''
                new_positive_summary_list.append(new_positive_summary)
            if args.negative_gen:
                new_negative_summary = ''
                new_negative_summary_list.append(new_negative_summary)                                                                                         

    split_dict = {
        'id': id_list,
        'prompt': new_prompt_list,
        'summary': new_summary_list,
        'topic': topic_list,
    }

    if args.postive_gen:
        split_dict['positive_prompt'] = new_positive_prompt_list
        split_dict['positive_topic'] = positive_topic_list
        if args.topic_prompt_output or args.length_prompt_output:
            split_dict['positive_summary'] = new_positive_summary_list
    
    if args.negative_gen:
        if args.negative_sample == 1:
            split_dict['negative_prompt'] = new_negative_prompt_list[0]
            split_dict['negative_topic'] = negative_topic_list[0]
            if args.topic_prompt_output or args.length_prompt_output:
                split_dict['negative_summary'] = new_negative_summary_list
        else:
            for num in range(args.negative_sample):
                key_prompt_name = 'negative_prompt_' + str(num)
                key_topic_name = 'negative_topic_' + str(num)
                split_dict[key_prompt_name] = [prompt[num] for prompt in new_negative_prompt_lists]
                split_dict[key_topic_name] = [topic[num] for topic in negative_topic_list]
                if args.topic_prompt_output or args.length_prompt_output:
                    key_summary_name = 'negative_summary_' + str(num)
                    split_dict[key_summary_name] = [summary[num] for summary in new_negative_summary_lists]

    split_dict = Dataset.from_dict(split_dict)

    return split_dict