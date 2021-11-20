import argparse
def get_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--train_file",
        default="data/conceptual_caption/training",
        type=str,
        # required=True,
        help="The input train corpus.",
    )
    parser.add_argument(
        "--validation_file",
        default="data/conceptual_caption/validation",
        type=str,
        # required=True,
        help="The input train corpus.",
    )
    parser.add_argument(
        "--from_pretrained",
        default="",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )
    parser.add_argument(
        "--bert_model",
        default="bert-base-uncased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )
    parser.add_argument(
        "--output_dir",
        default="save",
        type=str,
        # required=True,
        help="The output directory where the model checkpoints will be written.",
    )

    parser.add_argument(
        "--config_file",
        default="config/bert_config.json",
        type=str,
        # required=True,
        help="The config file which specified the model details.",
    )
    ## Other parameters
    parser.add_argument(
        "--max_seq_length",
        default=36,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. \n"
             "Sequences longer than this will be truncated, and sequences shorter \n"
             "than this will be padded.",
    )
    parser.add_argument("--predict_feature", action="store_true", help="visual target.")

    parser.add_argument(
        "--train_batch_size",
        default=512,
        type=int,
        help="Total batch size for training.",
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-4,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=10.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--start_epoch",
        default=0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for. "
             "E.g., 0.1 = 10%% of training.",
    )
    parser.add_argument("--caption_path",
                        type=str)
    parser.add_argument("--lmdb_file", type=str)
    parser.add_argument("--train_lmdb_file",type=str)
    parser.add_argument("--test_lmdb_file",type=str)
    parser.add_argument(
        "--img_weight", default=1, type=float, help="weight for image loss"
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Whether not to use CUDA when available"
    )
    parser.add_argument(
        "--on_memory",
        action="store_true",
        help="Whether to load train samples into memory or use disk",
    )
    parser.add_argument(
        "--do_lower_case",
        type=bool,
        default=True,
        help="Whether to lower case the input text. True for uncased models, False for cased models.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumualte before performing a backward/update pass.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit float precision instead of 32-bit",
    )
    parser.add_argument(
        "--loss_scale",
        type=float,
        default=0,
        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
             "0 (default value): dynamic loss scaling.\n"
             "Positive power of 2: static loss scaling value.\n",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=3,
        help="Number of workers in the dataloader.",
    )

    parser.add_argument(
        "--save_name",
        default='',
        type=str,
        help="save name for training.",
    )
    parser.add_argument(
        "--baseline", action="store_true", help="Wheter to use the baseline model (single bert)."
    )
    parser.add_argument(
        "--freeze", default=-1, type=int,
        help="till which layer of textual stream of vilbert need to fixed."
    )
    parser.add_argument(
        "--use_chuncks", default=0, type=float, help="whether use chunck for parallel training."
    )
    parser.add_argument(
        "--distributed", action="store_true", help="whether use chunck for parallel training."
    )
    parser.add_argument(
        "--without_coattention", action="store_true", help="whether pair loss."
    )

    parser.add_argument(
        "--label_list_file",type=str
    )

    parser.add_argument(
        "--MLM",action="store_true"
    )
    parser.add_argument(
        "--MRM",action="store_true"
    )
    parser.add_argument(
        "--ITM",action="store_true"
    )
    parser.add_argument(
        "--CLR",action="store_true"
    )

    args = parser.parse_args()

    return args







