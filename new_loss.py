import json
import math

from tqdm import tqdm
from collections import Counter

import nltk
from nltk.util import ngrams
from nltk import word_tokenize, sent_tokenize

from datasets import Dataset


def loss_topic(lprobs, target, epsilon, ignore_index=-100):
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