



import copy
import json
import logging
import os
import random

import lmdb
import numpy as np
import tensorpack.dataflow as td

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import torch.distributed as dist
import sys
import pdb

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def read_json(file):
    f=open(file,"r",encoding="utf-8").read()
    return json.loads(f)

def write_json(file,data):
    f=open(file,"w",encoding="utf-8")
    json.dump(data,f,indent=2,ensure_ascii=False)
    return


class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(
        self, image_feat=None,
            image_target=None,
            caption=None,
            is_next=None,
            lm_labels=None,
            image_loc=None,
            num_boxes=None
    ):
        self.image_feat = image_feat
        self.caption = caption
        self.is_next = is_next  # nextSentence
        self.lm_labels = lm_labels  # masked words for language model
        self.image_loc = image_loc
        self.image_target = image_target
        self.num_boxes = num_boxes

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
        self,
        input_ids=None,
        input_mask=None,
        segment_ids=None,
        is_next=None,
        lm_label_ids=None,
        image_feat=None,
        image_target=None,
        image_loc=None,
        image_label=None,
        image_mask=None
    ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.is_next = is_next
        self.lm_label_ids = lm_label_ids
        self.image_feat = image_feat
        self.image_loc = image_loc
        self.image_label = image_label
        self.image_target = image_target
        self.image_mask = image_mask


class Pretrain_DataSet_Train(object):
    def __init__(
            self,
            tokenizer,
            seq_len,
            predict_feature=False,
            batch_size=512,
            num_workers=25,
            lmdb_file=None,
            caption_path=None,
            MLM=True,
            MRM=True,
            ITM=True
        ):

        lmdb_file=lmdb_file
        caption_path=caption_path
        print("Loading from %s" % lmdb_file)

        ds = td.LMDBSerializer.load(lmdb_file, shuffle=False)
        self.num_dataset = len(ds)

        print("len: ",len(ds))

        preprocess_function = BertPreprocessBatch(
            caption_path,
            tokenizer,
            seq_len,
            36,
            predict_feature=predict_feature,
            MLM=MLM,
            MRM=MRM,
            ITM=ITM,
        )


        ds = td.MapData(ds, preprocess_function)
        self.ds = td.BatchData(ds, batch_size)
        self.ds.reset_state() # TODO: it is retained in the original version

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.MLM=MLM
        self.MRM=MRM
        self.ITM=ITM

    def __iter__(self):
        for batch in self.ds.get_data():
            input_ids, input_mask, segment_ids, lm_label_ids, is_next, \
            image_feat, image_loc, image_target, image_label, image_mask, \
            image_id = batch

            # image
            batch_size = input_ids.shape[0]
            g_image_feat = np.sum(image_feat, axis=1) / np.sum(image_mask, axis=1, keepdims=True)
            image_feat = np.concatenate([np.expand_dims(g_image_feat, axis=1), image_feat], axis=1)
            image_feat = np.array(image_feat, dtype=np.float32)

            g_image_loc = np.repeat(np.array([[0,0,1,1,1]], dtype=np.float32), batch_size, axis=0)
            image_loc = np.concatenate([np.expand_dims(g_image_loc, axis=1), image_loc], axis=1)
            image_loc = np.array(image_loc, dtype=np.float32)

            g_image_mask = np.repeat(np.array([[1]]), batch_size, axis=0)
            image_mask = np.concatenate([g_image_mask, image_mask], axis=1)

            batch = (input_ids, input_mask, segment_ids, lm_label_ids, is_next,
                     image_feat, image_loc, image_target, image_label, image_mask)

            yield tuple([torch.tensor(data) for data in batch]+ [image_id])

    def __len__(self):
        return self.ds.size()

class BertPreprocessBatch(object):
    def __init__(
        self,
        caption_path,
        tokenizer,
        seq_len,
        region_len,
        split="Train",
        predict_feature=False,
        visualization=False,
        MLM=True,
        MRM=True,
        ITM=True,
    ):

        self.MLM=MLM
        self.MRM=MRM
        self.ITM=ITM

        self.split = split
        self.seq_len = seq_len
        self.region_len = region_len
        self.tokenizer = tokenizer
        self.predict_feature = predict_feature

        self.id_info_dict=json.load(open(caption_path, 'r'))

        self.captions=[]
        for each in self.id_info_dict:
            self.captions.append(self.id_info_dict[each]["title"])
        self.num_caps=len(self.captions)
        self.visualization = visualization

    def __call__(self, data):

        image_feature_wp, image_location_wp, num_boxes,  image_h, image_w, image_id, caption = data
        
        image_feature = np.zeros((self.region_len, 2048), dtype=np.float32)
        image_target = np.zeros((self.region_len, 1601), dtype=np.float32)
        image_location = np.zeros((self.region_len, 5), dtype=np.float32)

        num_boxes = int(num_boxes)
        image_feature[:num_boxes] = image_feature_wp
        # image_target[:num_boxes] = image_target_wp
        image_location[:num_boxes,:4] = image_location_wp

        image_location[:,4] = (image_location[:,3] - image_location[:,1]) * (image_location[:,2] - image_location[:,0]) / (float(image_w) * float(image_h))
        
        image_location[:,0] = image_location[:,0] / float(image_w)
        image_location[:,1] = image_location[:,1] / float(image_h)
        image_location[:,2] = image_location[:,2] / float(image_w)
        image_location[:,3] = image_location[:,3] / float(image_h)

        if self.predict_feature:
            image_feature = copy.deepcopy(image_feature)
            image_target = copy.deepcopy(image_feature)
        else:
            image_feature = copy.deepcopy(image_feature)
            image_target = copy.deepcopy(image_target)


        # caption
        caption=caption
        caption, label = self.random_cap(caption)

        cur_example = InputExample(
            image_feat=image_feature,
            image_target=image_target,
            caption=caption,
            is_next=label,
            image_loc=image_location,
            num_boxes=num_boxes
        )

        # transform sample to features
        cur_features = self.convert_example_to_features(cur_example, self.seq_len, self.tokenizer, self.region_len)
        
        cur_tensors = (
            cur_features.input_ids,
            cur_features.input_mask,
            cur_features.segment_ids,
            cur_features.lm_label_ids,
            cur_features.is_next,
            cur_features.image_feat,
            cur_features.image_loc,
            cur_features.image_target,
            cur_features.image_label,
            cur_features.image_mask,
            image_id,
        )
        return cur_tensors

    def random_cap(self, caption):
        if self.visualization:
            return caption, 0

        if self.ITM:
            if random.random() > 0.5:
                label = 0
            else:
                caption = self.get_random_caption()
                label = 1
        else:
            label = 0

        return caption, label

    def get_random_caption(self):
        rand_doc_idx = random.randint(0, self.num_caps - 1)
        caption = self.captions[rand_doc_idx]

        return caption

    def convert_example_to_features(self, example, max_seq_length, tokenizer, max_region_length):
        image_feat = example.image_feat
        caption = example.caption
        image_loc = example.image_loc
        image_target = example.image_target
        num_boxes = int(example.num_boxes)

        caption = self.tokenizer.tokenize(caption)
        self._truncate_seq_pair(caption, max_seq_length - 2)

        # random mask
        caption, caption_label = self.random_word(caption, tokenizer)
        image_feat, image_loc, image_label = self.random_region(image_feat, image_loc, num_boxes)

        # concatenate lm labels
        lm_label_ids = [-1] + caption_label + [-1]
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in caption:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)


        if len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length]
            lm_label_ids = lm_label_ids[:max_seq_length]
            segment_ids=segment_ids[:max_seq_length]

        input_ids=tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * (len(input_ids))
        image_mask = [1] * (num_boxes)

        # Zero-pad up to the visual sequence length.
        while len(image_mask) < max_region_length:
            image_mask.append(0)
            image_label.append(-1)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            lm_label_ids.append(-1)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(lm_label_ids) == max_seq_length
        assert len(image_mask) == max_region_length
        assert len(image_label) == max_region_length

        features = InputFeatures(
            input_ids=np.array(input_ids),
            input_mask=np.array(input_mask),
            segment_ids=np.array(segment_ids),
            lm_label_ids=np.array(lm_label_ids),
            is_next=np.array(example.is_next),
            image_feat=image_feat,
            image_target=image_target,
            image_loc=image_loc,
            image_label=np.array(image_label),
            image_mask = np.array(image_mask)
        )
        return features

    def _truncate_seq_pair(self, tokens_b, max_length):
        while True:
            total_length = len(tokens_b)
            if total_length <= max_length:
                break

            tokens_b.pop()

    def random_word(self, tokens, tokenizer):
        output_label = []

        if self.MLM:
            for i, token in enumerate(tokens):
                prob = random.random()
                # mask token with 15% probability

                if prob < 0.15:
                    prob /= 0.15

                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        tokens[i] = "[MASK]"

                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]
                    # -> rest 10% randomly keep current token
                    # append current token to output (we will predict these later)
                    try:
                        output_label.append(tokenizer.vocab[token])
                    except KeyError:
                        # For unknown words (should not occur with BPE vocab)
                        output_label.append(tokenizer.vocab["[UNK]"])
                        logger.warning(
                            "Cannot find token '{}' in vocab. Using [UNK] insetad".format(token)
                        )
                else:
                    # no masking token (will be ignored by loss function later)
                    output_label.append(-1)
        else:
            for i, token in enumerate(tokens):
                output_label.append(-1)

        return tokens, output_label

    def random_region(self, image_feat, image_loc, num_boxes):
        """
        """
        output_label = []

        if self.MRM:
            for i in range(num_boxes):
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.15:
                    prob /= 0.15

                    # 80% randomly change token to mask token
                    if prob < 0.9:
                        image_feat[i] = 0
                    output_label.append(1)
                else:
                    # no masking token (will be ignored by loss function later)
                    output_label.append(-1)
        else:
            for i in range(num_boxes):
                output_label.append(-1)

        return image_feat, image_loc, output_label

