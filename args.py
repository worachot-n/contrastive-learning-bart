import argparse
from transformers import SchedulerType

def parse_args():
    arg_parser = argparse.ArgumentParser(description="bart")
    arg_parser.add_argument("--topic_prompt_input", dest="topic_prompt_input", type=bool,
                            default=False, help="Use topic prompt or not")
    arg_parser.add_argument("--length_prompt_input", dest="length_prompt_input", type=bool,
                            default=False, help="Use length prompt or not")
    arg_parser.add_argument("--predict_summary", dest="predict_summary", type=bool,
                            default=False, help="Use predict summary or not")
    arg_parser.add_argument("--output_dir", dest="output_dir",
                            type=str, default="./output/1", help="default")
    arg_parser.add_argument("--train_file", dest="train_file", type=str,
                            default=None, help="A json file containing the training data.")
    arg_parser.add_argument("--validation_file", dest="validation_file", type=str,
                            default=None, help="A json file containing the validation data.")
    arg_parser.add_argument("--test_file", dest="test_file", type=str,
                            default=None, help="A json file containing the test data.")
    arg_parser.add_argument("--ignore_pad_token_for_loss", dest="ignore_pad_token_for_loss", type=bool, default=True,
                            help="Whether to ignore the tokens corresponding to padded labels in the loss computation or not.",)
    arg_parser.add_argument("--text_column", dest="text_column", type=str, default="dialogue",
                            help="The name of the column in the datasets containing the full texts (for summarization).")
    arg_parser.add_argument("--summary_column", dest="summary_column", type=str, default="summary",
                            help="The name of the column in the datasets containing the summaries (for summarization).")
    arg_parser.add_argument("--model_name_or_path", dest="model_name_or_path", type=str, default="facebook/bart-large",
                            help="Path to pretrained model or model identifier from huggingface.co/models.")
    arg_parser.add_argument("--model_type", dest="model_type", type=str, default="bart",
                            help="Model type to use if training from scratch.")
    arg_parser.add_argument("--max_source_length", dest="max_source_length", 
                            type=int, default=1024, help="default")
    arg_parser.add_argument("--preprocessing_num_workers", type=int, default=None,
                            help="The number of processes to use for the preprocessing.")
    arg_parser.add_argument("--overwrite_cache", dest="overwrite_cache", type=bool,
                            default=None, help="Overwrite the cached training and evaluation sets")
    arg_parser.add_argument("--min_target_length", dest="min_target_length", type=int,
                            default=1, help="The minimal total sequence length for target text")
    arg_parser.add_argument("--max_target_length", dest="max_target_length", type=int, default=128, help="The maximum total sequence length for target text"
                            "after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded. during ``evaluate`` and"
                            "``predict``.")
    arg_parser.add_argument("--num_beams", dest="num_beams", type=int, default=4, help="Number of beams to use for evaluation. This argument will be "
                            "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.")
    arg_parser.add_argument("--learning_rate", dest="learning_rate", type=float, default=5e-5,
                            help="Initial learning rate (after the potential warmup period) to use.")
    arg_parser.add_argument("--pad_to_max_length", action="store_true",
                            help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",)
    arg_parser.add_argument("--weight_decay", dest="weight_decay",
                            type=float, default=1e-3, help="Weight decay to use.")
    arg_parser.add_argument("--label_smoothing", dest="label_smoothing",
                            type=float, default=0.1, help="hyperparameter for label smoothing.")
    arg_parser.add_argument("--length_penalty", dest="length_penalty", type=float,
                            default=1.0, help="large - longer sequence, small - shorter sequence")
    arg_parser.add_argument("--num_train_epochs", dest="num_train_epochs",
                            type=int, default=15, help="Total number of training epochs to perform.")
    arg_parser.add_argument("--per_device_train_batch_size", dest="per_device_train_batch_size",
                            type=int, default=8, help="Batch size (per device) for the training dataloader.")
    arg_parser.add_argument("--gradient_accumulation_steps", dest="gradient_accumulation_steps", type=int,
                            default=64, help="Number of updates steps to accumulate before performing a backward/update pass.")
    arg_parser.add_argument("--per_device_eval_batch_size", dest="per_device_eval_batch_size",
                            type=int, default=8, help="Batch size (per device) for the evaluation dataloader.")
    arg_parser.add_argument("--per_device_test_batch_size", dest="per_device_test_batch_size",
                            type=int, default=8, help="Batch size (per device) for the evaluation dataloader.")
    arg_parser.add_argument("--num_warmup_steps", dest="num_warmup_steps", type=int,
                            default=0, help="Number of steps for the warmup in the lr scheduler.")
    arg_parser.add_argument("--cache_dir", dest="cache_dir",
                            type=str, default="./output/cache", help="default")
    arg_parser.add_argument("--seed", dest="seed",
                            type=int, default=12345, help="default")
    arg_parser.add_argument("--config_name", type=str, default=None,
                            help="Pretrained config name or path if not the same as model_name")
    arg_parser.add_argument("--tokenizer_name", type=str, default=None,
                            help="Pretrained tokenizer name or path if not the same as model_name")
    arg_parser.add_argument("--use_slow_tokenizer", dest="use_slow_tokenizer", action="store_true",
                            help="If passed, will use a slow tokenizer (not backed by the HuggingFaceTokenizers library).")
    arg_parser.add_argument("--max_train_steps", type=int, default=None,
                            help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    arg_parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear", help="The scheduler type to use.",
                            choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    arg_parser.add_argument("--embedding_lr", type=float, default=5e-5,
                            help="Initial learning rate for embedding layers.")
    arg_parser.add_argument("--len_start", type=int,
                            default=1, help="start length.")
    arg_parser.add_argument("--len_end", type=int,
                            default=100, help="end length.")
    arg_parser.add_argument("--contrastive_loss", dest="contrastive_loss", type=bool,
                            default=False, help="Use contrastive loss or not")
    arg_parser.add_argument("--tagging", dest="tagging", type=str, default="no",
                            choices=('no', 'word', 'prompt'), help="Use tagging (<tp>, </tp>) in word, sentence, or not")
    arg_parser.add_argument("--synonym_replacement", dest="synonym_replacement", type=bool,
                            default=False, help="Synonym replacement or not")
    arg_parser.add_argument("--random_topic", dest="random_topic", type=bool,
                            default=False, help="Random topic or not")
    arg_parser.add_argument("--contrastive_encoder", dest="contrastive_encoder", type=bool,
                            default=False, help="Contrastive encoder or not")
    arg_parser.add_argument("--gen_sample", dest="gen_sample", type=int,
                            default=1, help="The number of sample")
    arg_parser.add_argument("--alpha", dest="alpha", type=float,
                            default=0.5, help="ration of computation loss in encoder")
    arg_parser.add_argument("--margin", dest="margin", type=float,
                            default=0.5, help="margin of computation loss")
    arg_parser.add_argument("--debug", action='store_true',
                            default=False, help="Use the debug mode or not")
    args = arg_parser.parse_args()

    # Sanity checks
    if args.train_file is None and args.validation_file is None:
        raise ValueError(
            "Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["json", "jsonl"], "`train_file` should be a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["json", "jsonl"], "`validation_file` should be a json file."

    return args