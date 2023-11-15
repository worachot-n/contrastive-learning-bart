import math
import os
import pprint
import logging

import datasets
import nltk
import numpy as np
import torch
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from filelock import FileLock
from transformers import AdamW, get_scheduler, set_seed

from transformers.file_utils import is_offline_mode
from transformers.utils.versions import require_version

from args import parse_args
from data_loader import raw_data_loader, data_processor
from model_loader import model_loader
from rouge_s import py_rouge_scores
from utils import label_smoothed_nll_loss, postprocess_text
from contrastive_loss import cosine_embedding_loss, margin_ranking_loss


# =  =  =  =  =  =  =  =  =  = Logging Setup =  =  =  =  =  =  =  =  =  =  =  = 

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# =  =  =  =  =  =  =  =  =  = Pre-check Package Info =  =  =  =  =  =  =  =  =  =  =  = 
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

# = = = = = = = = = = = = = Main Process = = = = = = = = = = = = = = = = = =
def main():
    args = parse_args()
    
    # display parameters
    logging.info("*** Parameters ***")
    for item, value in vars(args).items():
        logging.info("{}: {}".format(item, value))
    logging.info("")

    # Initialize the accelerator. The accelerator will handle device placement for us.
    accelerator = Accelerator(mixed_precision="fp16")
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        torch.backends.cudnn.enabled = False 
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # load raw dataset
    raw_datasets = raw_data_loader(args)

    # load model (config, tokenizer, s2s model)
    config, tokenizer, model = model_loader(accelerator, logger, args)
    
    # data processor (for DataLoader)
    dataloader, processed_dataset = data_processor(logger, args, accelerator, raw_datasets, tokenizer, model)
    train_dataloader, eval_dataloader, test_dataloader = dataloader
    train_dataset, _, _ = processed_dataset

    # = = = Training Preparation = = =
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    no_decay_emb_matrix = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay_emb_matrix)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    # optimizer
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    model, optimizer, train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, test_dataloader
    )

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # = = = = = = = = = = = = = = = = Train = = = = = = = = = = = = = = = = = = =
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f" Num examples = {len(train_dataset)}")
    logger.info(f" Num Epochs = {args.num_train_epochs}")
    logger.info(f" Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f" Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f" Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f" Total optimization steps = {args.max_train_steps}")
    
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), desc="Training: ", disable=not accelerator.is_local_main_process)
    completed_steps = 0

    val_results = []
    acc_losses  = []
    best_r2_f1  = None
    best_epoch  = 0
    
    if args.model_type == 'bart' or args.model_type == 't5':
        task_specific_params = model.config.task_specific_params
        params = task_specific_params.get('summarization', {})
        params['min_length'] = args.min_target_length
        params['max_length'] = args.max_target_length
        params['length_penalty'] = args.length_penalty
        params['num_beams'] = args.num_beams
        model.config.update(params)
    else:
        raise ValueError('{} model type not implemented'.format(args.model_type))
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = Train =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 
    for epoch in range(args.num_train_epochs):
        # train
        model.train()
        for step, batch in enumerate(train_dataloader):
            if args.label_smoothing == 0:
                outputs = model(**batch)
                loss = outputs.loss
            else:
                outputs = model(**batch, output_hidden_states=True)
                output_logits = outputs.logits

                output_probs = torch.nn.functional.log_softmax(
                    output_logits, dim=-1)

                if args.contrastive_loss:
                    max_encoder_token = model.config.max_position_embeddings
                    # print(max_encoder_token)

                    divide_num = int(output_probs.shape[0] / 2)
                    # print(divide_num)

                    embeddings_1 = outputs.encoder_last_hidden_state[0,:,:max_encoder_token]
                    synonym_embeddings_1 = outputs.encoder_last_hidden_state[2,:,:max_encoder_token]
                    random_embeddings_1 = outputs.encoder_last_hidden_state[4,:,:max_encoder_token]
                    # synonym_embeddings = synonym_embeddings.view(-1, max_encoder_token)
                    # random_embeddings = random_embeddings.view(-1, max_encoder_token)
                    synonym_1 = -1 * torch.ones(synonym_embeddings_1.size(dim=0)).to(device)
                    random_1 = -1 * torch.ones(random_embeddings_1.size(dim=0)).to(device)
                    embeddings_2 = outputs.encoder_last_hidden_state[1,:,:max_encoder_token]
                    synonym_embeddings_2 = outputs.encoder_last_hidden_state[3,:,:max_encoder_token]
                    random_embeddings_2 = outputs.encoder_last_hidden_state[5,:,:max_encoder_token]
                    synonym_2 = -1 * torch.ones(synonym_embeddings_2.size(dim=0)).to(device)
                    random_2 = -1 * torch.ones(random_embeddings_2.size(dim=0)).to(device)
                    # print(embeddings_1.shape)
                    # print(synonym_embeddings_1.shape)
                    # print(random_embeddings_1.shape)
                    # print(embeddings.shape)
                    # print(synonym_embeddings_2.shape)
                    # print(random_embeddings_2.shape)
                    # break

                    loss_cs_synonym_1 = cosine_embedding_loss(embeddings_1, synonym_embeddings_1, synonym_1, args.margin)
                    loss_cs_random_1 = cosine_embedding_loss(embeddings_1, random_embeddings_1, random_1, args.margin)
                    loss_cs_synonym_2 = cosine_embedding_loss(embeddings_2, synonym_embeddings_2, synonym_2, args.margin)
                    loss_cs_random_2 = cosine_embedding_loss(embeddings_2, random_embeddings_2, random_2, args.margin)
                    # loss_cs_1 = loss_cs_synonym_1 + loss_cs_random_1
                    # loss_cs_2 = loss_cs_synonym_2 + loss_cs_random_2
                    # loss_cs = (loss_cs_1 + loss_cs_2) / 2
                    # loss_cs_synonym = (loss_cs_synonym_1 + loss_cs_synonym_2) / 2
                    # loss_cs_random = (loss_cs_random_1 + loss_cs_random_2) / 2
                    loss_cs = (loss_cs_synonym_1 + loss_cs_synonym_2 + loss_cs_random_1 + loss_cs_random_2) / 4
                    # print(f"loss_cs: {loss_cs}")

                    
                    output_probs_1 = output_probs[0,:,:]
                    # print(output_probs_1.shape)
                    output_probs_2 = output_probs[1,:,:]
                    # print(output_probs_2.shape)
                    output_probs_all = torch.stack((output_probs_1, output_probs_2), dim=1)
                    # print("output_probs_all: ", output_probs_all.shape)
                    
                    # ## decoder
                    # output_probs_synonym_1 = output_probs[2,:,:]
                    # output_probs_synonym_2 = output_probs[4,:,:]
                    # output_probs_synonym = torch.stack((output_probs_synonym_1, output_probs_synonym_2), dim=1)
                    # # print("output_probs_synonym: ", output_probs_synonym.shape)
                    # output_probs_random_1 = output_probs[3,:,:]
                    # output_probs_random_2 = output_probs[5,:,:]
                    # output_probs_random = torch.stack((output_probs_random_1, output_probs_random_2), dim=1)
                    # # print("output_probs_random: ", output_probs_random.shape)
                    # output_probs_all_mr = output_probs_all.view(-1,
                    #                                  model.config.vocab_size)
                    # # print("output_probs_all_mr: ", output_probs_all_mr.shape)
                    # output_probs_synonym = output_probs_synonym.view(-1,
                    #                                  model.config.vocab_size)
                    # # print("output_probs_synonym: ", output_probs_synonym.shape)
                    # output_probs_random = output_probs_random.view(-1,
                    #                                  model.config.vocab_size)
                    # # print("output_probs_random: ", output_probs_random.shape)
                    # # (pos, neg, target, ignore_index=-100, ,device)
                    # target_one = torch.ones(gt_logits_all_mr.shape[0]).to(device)
                    # # print("target_one: ", target_one.shape)
                    # loss_mr_1 = margin_ranking_loss(output_probs_all_mr, output_probs_synonym, 
                    #                                           gt_logits_all_mr, target_one, ignore_index=tokenizer.pad_token_id)
                    # loss_mr_2 = margin_ranking_loss(output_probs_all_mr, output_probs_random, 
                    #                                           gt_logits_all_mr, target_one, ignore_index=tokenizer.pad_token_id)
                    # loss_mr = (loss_mr_1 + loss_mr_2) / 2
                    # # print(f"loss_mr: {loss_mr}")
                    
                                        
                    ## negative log-likelihood
                        
                    # gt_logits = batch['labels'][:divide_num]
                    gt_logits = batch['labels']
                    # print("gt_logits: ", gt_logits.shape)
                    # gt_logits = gt_logits.view(-1)
                    gt_logits_1 = gt_logits[0,:]
                    gt_logits_2 = gt_logits[1,:]
                    gt_logits_all = torch.stack((gt_logits_1, gt_logits_2), dim=1)
                    
                    # ## decoder
                    # gt_logits_all_mr = gt_logits_all.view(-1)
                    # # print("gt_logits_all_mr: ", gt_logits_all_mr.shape)

                    loss_nll, nll = label_smoothed_nll_loss(
                        output_probs_all, gt_logits_all, args.label_smoothing, ignore_index=tokenizer.pad_token_id)

                    loss = loss_nll + (args.alpha * loss_cs)
                    
                    # ## decoder
                    # loss = loss_nll + (args.alpha * loss_cs) + (args.beta * loss_mr)
                    # print(loss)
                    # break

                else:
                    output_probs = output_probs
                    output_probs = output_probs.view(-1,
                                                     model.config.vocab_size)

                    gt_logits = batch['labels']
                    gt_logits = gt_logits.view(-1)

                    loss_nll, nll = label_smoothed_nll_loss(
                        output_probs, gt_logits, args.label_smoothing, ignore_index=tokenizer.pad_token_id)

                    loss = loss_nll
                    # break

            acc_losses.append(loss.item())
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)

            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                progress_bar.set_postfix(lr=lr_scheduler.get_last_lr()[0], loss=np.mean(acc_losses[-50:]))
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break

        # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = EVAL =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 
        model.eval()
        val_predict     = []
        val_groundtruth = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )

                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )
                labels = batch["labels"]
                if not args.pad_to_max_length:
                    # If we did not pad to max length, we need to pad the labels too
                    labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

                generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                labels = accelerator.gather(labels).cpu().numpy()

                if args.ignore_pad_token_for_loss:
                    # Replace -100 in the labels as we can't decode them.
                    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]

                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

                val_predict.extend(decoded_preds)
                val_groundtruth.extend(decoded_labels)

        # if args.topic_prompt_output:
        #     new_val_predict = []
        #     new_val_groundtruth = []
        #     for sample_predict, smaple_groundtruth in zip(val_predict, val_groundtruth):
        #         try:
        #             gen_sum = sample_predict.split('Summary: ')[1]
        #             new_val_predict.append(gen_sum)
        #         except:
        #             new_val_predict.append(sample_predict)
        #         truth_sum = smaple_groundtruth.split('Summary: ')[1]
        #         new_val_groundtruth.append(truth_sum)
        #     val_predict = new_val_predict
        #     val_groundtruth = new_val_groundtruth

        logger.info("")
        logger.info("Rouge score on val set after epoch {}".format(epoch+1))
        eval_results = py_rouge_scores(val_predict, val_groundtruth)

        if best_r2_f1 is None:
            best_r2_f1 = eval_results
        if eval_results['rouge-2']['f'] >= best_r2_f1['rouge-2']['f']:
            best_r2_f1 = eval_results
            best_epoch = epoch + 1

            os.makedirs(args.output_dir+'/best', exist_ok=True)
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(args.output_dir+'/best', save_function=accelerator.save)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir+'/best')

            # save vocab
            vocab = tokenizer.vocab.copy()
            vocab = {k: v for k, v in sorted(vocab.items(), key=lambda item: item[1])}
            with open(args.output_dir + '/best/vocab.txt', 'w') as f:
                for word, index in vocab.items():
                    # it lead to encoding bug on some machines, so i add this line
                    word = word.encode('ascii', 'ignore').decode('ascii')
                    f.write(str(index) + ': ' + word + '\n')

        # = = = = = = = = = = = = = = = = = = = = = = = = =
        logger.info("Current Best Validation Result is at epoch {}".format(best_epoch))
        py_rouge_scores(None, None, best_r2_f1)


    # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = Test =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 
    # load best model
    logger.info("Loading Best Result is at epoch {} for Testing".format(best_epoch))

    unwrapped_model = accelerator.unwrap_model(model)
    config          = config.from_pretrained(args.output_dir+'/best')
    tokenizer       = tokenizer.from_pretrained(args.output_dir+'/best', config=config)
    unwrapped_model = unwrapped_model.from_pretrained(args.output_dir+'/best', config=config)
    model           = accelerator.prepare(unwrapped_model)

    if args.model_type == 'bart' or args.model_type == 't5':
        task_specific_params = model.config.task_specific_params
        params = task_specific_params.get('summarization', {})
        params['min_length'] = args.min_target_length
        params['max_length'] = args.max_target_length
        params['length_penalty'] = args.length_penalty
        params['num_beams'] = args.num_beams
        model.config.update(params)
    else:
        raise ValueError('{} model type not implemented'.format(args.model_type))

    # start Test 
    logger.info("Collecting Testing Result...")
    model.eval()

    test_predict     = []
    test_groundtruth = []
    for step, batch in enumerate(tqdm(test_dataloader, leave=False)):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            labels = batch["labels"]

            if not args.pad_to_max_length:
                # If we did not pad to max length, we need to pad the labels too
                labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
            labels = accelerator.gather(labels).cpu().numpy()

            if args.ignore_pad_token_for_loss:
                # Replace -100 in the labels as we can't decode them.
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]

            decoded_preds  = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

            decoded_preds  = [' '.join(sent.split('\n')) for sent in decoded_preds]
            decoded_labels = [' '.join(sent.split('\n')) for sent in decoded_labels]

            test_predict.extend(decoded_preds)
            test_groundtruth.extend(decoded_labels)

    # if args.topic_prompt_output:
    #     new_test_predict = []
    #     new_test_groundtruth = []
    #     for sample_predict, smaple_groundtruth in zip(test_predict, test_groundtruth):
    #         try:
    #             gen_sum = sample_predict.split('Summary: ')[1]
    #             new_test_predict.append(gen_sum)
    #         except:
    #             new_test_predict.append(sample_predict)
    #             new_test_groundtruth.append(smaple_groundtruth)
    #         truth_sum = smaple_groundtruth.split('Summary: ')[1]
    #         new_test_groundtruth.append(truth_sum)
    #     test_predict = new_test_predict
    #     test_groundtruth = new_test_groundtruth
    
    print(raw_datasets['test']['prompt'][0])

    logger.info("")
    logger.info("ROUGE score on test set")
    test_scores = py_rouge_scores(test_predict, test_groundtruth)
    logger.info("")


    # Save generated summaries
    if args.predict_summary:
        os.makedirs(args.output_dir+'/predict_gen_samples', exist_ok=True)
    else:
        os.makedirs(args.output_dir+'/gen_samples', exist_ok=True)

    for i in range(len(test_predict)):
        test_id        = raw_datasets['test']['id'][i]
        test_dialogue  = raw_datasets['test']['prompt'][i]
        test_summary   = raw_datasets['test']['summary'][i]
        test_predict_s = test_predict[i]

        if args.predict_summary:
            with open(args.output_dir+'/predict_gen_samples/'+str(test_id)+'.txt', 'w') as f:
                test_dialogue = test_dialogue.encode('ascii', 'ignore').decode('ascii')
                f.write(test_dialogue)
                f.write('\n\n')
                f.write('Golden Summary:\n')
                test_summary = test_summary.encode('ascii', 'ignore').decode('ascii')
                f.write(test_summary)
                f.write('\n\n')
                f.write('Generate Summary:\n')
                test_predict_s = test_predict_s.encode('ascii', 'ignore').decode('ascii')
                f.write(test_predict_s)
        else:
            with open(args.output_dir+'/gen_samples/'+str(test_id)+'.txt', 'w') as f:
                test_dialogue = test_dialogue.encode('ascii', 'ignore').decode('ascii')
                f.write(test_dialogue)
                f.write('\n\n')
                f.write('Golden Summary:\n')
                test_summary = test_summary.encode('ascii', 'ignore').decode('ascii')
                f.write(test_summary)
                f.write('\n\n')
                f.write('Generate Summary:\n')
                test_predict_s = test_predict_s.encode('ascii', 'ignore').decode('ascii')
                f.write(test_predict_s)


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# main process
if __name__ == "__main__":
    main()
