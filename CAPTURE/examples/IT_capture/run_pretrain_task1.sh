#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=4 python pretrain_task.py \
 --from_pretrained /data1/xl/bert_model/bert_base_chinese \
 --bert_model bert-base-chinese \
 --config_file ../../config/capture.json\
 --predict_feature \
 --learning_rate 1e-4 \
 --train_batch_size 64 \
 --max_seq_length 36 \
 --lmdb_file ${data_root}/lmdb_features/train.lmdb \
 --caption_path ${data_root}/train_id_info.json \
 --save_name capture_subset_v2_MLM_MRM_CLR \
 --MLM \
 --MRM \
 --CLR