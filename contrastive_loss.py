import json
import math

from tqdm import tqdm
import torch.nn as nn
from collections import Counter

import nltk
from nltk.util import ngrams
from nltk import word_tokenize, sent_tokenize

from datasets import Dataset

mr_loss = nn.MarginRankingLoss()

def cosine_embedding_loss(pos, neg, contrastive, margin=0.5):
    cs_loss = nn.CosineEmbeddingLoss(margin)
    loss_cosine_embedding = cs_loss(pos, neg, contrastive)
    return loss_cosine_embedding

def margin_ranking_loss(pos, neg, target, target_one, ignore_index=-100):
    
    probs_pos = pos[~target.eq(-100)]
    probs_neg = neg[~target.eq(-100)]
    target = target[~target.eq(-100)]
    target_one = target_one[:target.shape[0]]

    if target.dim() == probs_pos.dim() - 1:
        target = target.unsqueeze(-1)
    
    nll_pos = -probs_pos.gather(dim=-1, index=target)
    nll_neg = -probs_neg.gather(dim=-1, index=target)

    nll_sq_pos = nll_pos.squeeze(-1)
    nll_sq_neg = nll_neg.squeeze(-1)

    loss_margin_ranking = mr_loss(nll_sq_pos, nll_sq_neg, target_one)

    return loss_margin_ranking