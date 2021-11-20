import argparse
import json
import logging
import os
import random
from io import open
import math
import sys
sys.path.append("../../")

from time import gmtime, strftime
from timeit import default_timer as timer

import numpy as np
from tqdm import tqdm, trange

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from dataloaders.pretrain_dataset_IT import Pretrain_DataSet_Train
from model.capture_IT import BertForMultiModalPreTraining, BertConfig

import torch.distributed as dist

import pdb
from utils_args import get_args
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():

    args = get_args()


    print(args)
    if args.save_name is not '':
        timeStamp = args.save_name
    else:
        timeStamp = strftime("%d-%b-%y-%X-%a", gmtime())
        timeStamp += "_{:0>6d}".format(random.randint(0, 10e6))

    savePath = os.path.join(args.output_dir, timeStamp)

    if not os.path.exists(savePath):
        os.makedirs(savePath)
    
    config = BertConfig.from_json_file(args.config_file)
    
    if args.freeze > config.t_biattention_id[0]:
        config.fixed_t_layer = config.t_biattention_id[0]

    if args.without_coattention:
        config.with_coattention = False
    # save all the hidden parameters. 
    with open(os.path.join(savePath, 'command.txt'), 'w') as f:
        print(args, file=f)  # Python 3.x
        print('\n', file=f)
        print(config, file=f)

    bert_weight_name = json.load(open("../../config/bert-base-uncased_weight_name.json", "r"))
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend="nccl")
    logger.info(
        "device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, n_gpu, bool(args.local_rank != -1), args.fp16
        )
    )

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps
            )
        )

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case
    )

    viz = TBlogger("logs", timeStamp)

    print("MLM: {} MRM:{} ITM:{}, CLR:{}".format(args.MLM, args.MRM, args.ITM,args.CLR))


    train_dataset = Pretrain_DataSet_Train(
        tokenizer,
        seq_len=args.max_seq_length,
        batch_size=args.train_batch_size,
        predict_feature=args.predict_feature,
        num_workers=args.num_workers,
        lmdb_file=args.lmdb_file,
        caption_path=args.caption_path,
        MLM=args.MLM,
        MRM=args.MRM,
        ITM=args.ITM
    )

    num_train_optimization_steps = (
        int(
            train_dataset.num_dataset
            / args.train_batch_size
            / args.gradient_accumulation_steps
        )
        * (args.num_train_epochs - args.start_epoch)
    )

    default_gpu = False
    if dist.is_available() and args.distributed:
        rank = dist.get_rank()
        if rank == 0:
            default_gpu = True
    else:
        default_gpu = True

    # pdb.set_trace()
    if args.predict_feature:
        config.v_target_size = 2048
        config.predict_feature = True
    else:
        config.v_target_size = 1601
        config.predict_feature = False

    if args.from_pretrained:
        model = BertForMultiModalPreTraining.from_pretrained(args.from_pretrained, config)
    else:
        model = BertForMultiModalPreTraining(config)

    model.cuda()

    if args.fp16:
        model.half()
    if args.local_rank != -1:
        try:
            from apex import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )
        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    if args.freeze != -1:
        bert_weight_name_filtered = []
        for name in bert_weight_name:
            if 'embeddings' in name:
                bert_weight_name_filtered.append(name)
            elif 'encoder' in name:
                layer_num = name.split('.')[2]
                if int(layer_num) <= args.freeze:
                    bert_weight_name_filtered.append(name)

        optimizer_grouped_parameters = []
        for key, value in dict(model.named_parameters()).items():
            if key[12:] in bert_weight_name_filtered:
                value.requires_grad = False

        if default_gpu:
            print("filtered weight")
            print(bert_weight_name_filtered)

    if not args.from_pretrained:
        param_optimizer = list(model.named_parameters())
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
    else:
        optimizer_grouped_parameters = []
        for key, value in dict(model.named_parameters()).items():
            if value.requires_grad:
                if key[12:] in bert_weight_name:
                    lr = args.learning_rate * 0.1
                else:
                    lr = args.learning_rate

                if any(nd in key for nd in no_decay):
                    optimizer_grouped_parameters += [
                        {"params": [value], "lr": lr, "weight_decay": 0.01}
                    ]

                if not any(nd in key for nd in no_decay):
                    optimizer_grouped_parameters += [
                        {"params": [value], "lr": lr, "weight_decay": 0.0}
                    ]
        if default_gpu:
            print(len(list(model.named_parameters())), len(optimizer_grouped_parameters))

    # set different parameters for vision branch and lanugage branch.
    if args.fp16:
        try:
            from apex import FP16_Optimizer
            from apex import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )

        optimizer = FusedAdam(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            bias_correction=False,
            max_grad_norm=1.0,
        )
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        if args.from_pretrained:
            optimizer = BertAdam(
                optimizer_grouped_parameters,
                warmup=args.warmup_proportion,
                t_total=num_train_optimization_steps,

            )

        else:
            optimizer = BertAdam(
                optimizer_grouped_parameters,
                lr=args.learning_rate,
                warmup=args.warmup_proportion,
                t_total=num_train_optimization_steps,
            )

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", train_dataset.num_dataset)
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    startIterID = 0
    global_step = 0
    masked_loss_v_tmp = 0
    masked_loss_t_tmp = 0
    next_sentence_loss_tmp = 0
    loss_tmp = 0
    start_t = timer()


    print("Prepare to training!")
    print("MLM: {} MRM:{} ITM:{}".format(args.MLM, args.MRM, args.ITM))
    right_temp=0
    all_num_temp=0


    for epochId in range(int(args.start_epoch), int(args.num_train_epochs)):
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataset)):
            iterId = startIterID + step + (epochId * len(train_dataset))
            image_id=batch[-1]
            batch=batch[:-1]
            batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)

            input_ids, input_mask, segment_ids, lm_label_ids, is_next,\
            image_feat, image_loc, image_target, image_label, image_mask= (
                batch
            )

            masked_loss_t,masked_loss_v, \
            next_sentence_loss = model(
                input_ids,
                image_feat,
                image_loc,
                segment_ids,
                input_mask,
                image_mask,
                lm_label_ids,
                image_label,
                image_target,
                is_next,
            )

            masked_loss_v = masked_loss_v * args.img_weight
            if not args.MLM:
                masked_loss_t=masked_loss_t*0
            if not args.MRM:
                masked_loss_v=masked_loss_v*0
            if not args.CLR:
                next_sentence_loss=next_sentence_loss*0

            loss = masked_loss_t \
                   + masked_loss_v \
                   + next_sentence_loss

            right=0
            all_num=0
            right_temp+=right
            all_num_temp+=all_num

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
                masked_loss_t = masked_loss_t.mean()
                masked_loss_v = masked_loss_v.mean()
                next_sentence_loss = next_sentence_loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            if math.isnan(loss.item()):
                pdb.set_trace()

            tr_loss += loss.item()

            rank = 0

            if dist.is_available() and args.distributed:
                rank = dist.get_rank()
            else:
                rank = 0
                
            viz.linePlot(iterId, loss.item(), "loss_"+str(rank), "train")
            viz.linePlot(iterId, masked_loss_t.item(), "masked_loss_t_"+str(rank), "train")
            viz.linePlot(iterId, masked_loss_v.item(), "masked_loss_v_"+str(rank), "train")
            viz.linePlot(
                iterId, next_sentence_loss.item(), "next_sentence_loss_"+str(rank), "train"
            )
            # viz.linePlot(iterId, optimizer.get_lr()[0], 'learning_rate', 'train')

            loss_tmp += loss.item()
            masked_loss_v_tmp += masked_loss_v.item()
            masked_loss_t_tmp += masked_loss_t.item()
            next_sentence_loss_tmp += next_sentence_loss.item()

            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    # modify learning rate with special warm up BERT uses
                    # if args.fp16 is False, BertAdam is used that handles this automatically
                    lr_this_step = args.learning_rate * warmup_linear(
                        global_step / num_train_optimization_steps,
                        args.warmup_proportion,
                    )
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr_this_step

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if step % 20 == 0 and step != 0:
                masked_loss_t_tmp = masked_loss_t_tmp / 20.0
                masked_loss_v_tmp = masked_loss_v_tmp / 20.0
                next_sentence_loss_tmp = next_sentence_loss_tmp / 20.0
                loss_tmp = loss_tmp / 20.0

                end_t = timer()
                timeStamp = strftime("%a %d %b %y %X", gmtime())

                Ep = epochId + nb_tr_steps / float(len(train_dataset))
                printFormat = "[%s][Ep: %.2f][Iter: %d][Time: %5.2fs][Loss: %.5g][Loss_v: %.5g][Loss_t: %.5g][Loss_n: %.5g][LR: %.8g][epoch: %d][step: %d]"

                printInfo = [
                    timeStamp,
                    Ep,
                    nb_tr_steps,
                    end_t - start_t,
                    loss_tmp,
                    masked_loss_v_tmp,
                    masked_loss_t_tmp,
                    next_sentence_loss_tmp,
                    optimizer.get_lr()[0],
                    epochId,
                    step
                ]
                
                start_t = end_t
                print(printFormat % tuple(printInfo))

                masked_loss_v_tmp = 0
                masked_loss_t_tmp = 0
                next_sentence_loss_tmp = 0
                loss_tmp = 0

        if default_gpu:
            # Save a trained model
            logger.info("** ** * Saving fine - tuned model ** ** * ")
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Only save the model it-self
            output_model_file = os.path.join(
                savePath, "pytorch_model_" + str(epochId) + ".bin"
            )

            torch.save(model_to_save.state_dict(), output_model_file)

class TBlogger:
    def __init__(self, log_dir, exp_name):
        log_dir = log_dir + "/" + exp_name
        print("logging file at: " + log_dir)
        self.logger = SummaryWriter(log_dir=log_dir)

    def linePlot(self, step, val, split, key, xlabel="None"):
        self.logger.add_scalar(split + "/" + key, val, step)

if __name__ == "__main__":
    # import time
    # time.sleep(3600*3)

    main()
