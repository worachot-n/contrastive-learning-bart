import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


from utils import label_smoothed_nll_loss


class ContrastiveModel(nn.Module):

    def __init__(self, args, config):
        '''initialization'''
        super(ContrastiveModel, self).__init__()

        self.seq2seq_model = AutoModelForSeq2SeqLM.from_pretrained(
                    args.model_name_or_path,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                    cache_dir=args.cache_dir,
        )

        self.args            = args
        self.config          = self.seq2seq_model.config
        self.generate        = self.seq2seq_model.generate
        self.from_pretrained = self.seq2seq_model.from_pretrained
        self.save_pretrained = self.seq2seq_model.save_pretrained

        self.sim_loss        = args.sim_loss
        self.label_smoothing = args.label_smoothing


    def forward(self, batch, tokenizer):
        '''
            batch computation
        '''

        negative_batch = batch['negative_input_ids']
        outputs = self.seq2seq_model(**batch,output_hidden_states=True)
        
        # label smoothing loss for addtional embeddings
        if not self.label_smoothing:
            loss = outputs.loss
        else:
            output_logits = outputs.logits
            output_probs = torch.nn.functional.log_softmax(output_logits, dim=-1)
            output_probs = output_probs.view(-1, self.config.vocab_size)


            gt_logits = batch['labels']
            gt_logits = gt_logits.view(-1)

            loss, _ = label_smoothed_nll_loss(output_probs, gt_logits, self.label_smoothing, ignore_index=tokenizer.pad_token_id)

        return outputs, loss, negative_batch