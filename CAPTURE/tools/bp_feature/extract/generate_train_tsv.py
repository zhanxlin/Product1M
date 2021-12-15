import argparse
import pprint
import time, os, sys
import base64
import numpy as np
import cv2
import csv
csv.field_size_limit(sys.maxsize)
from multiprocessing import Process
import random
import json
import pdb
import pandas as pd
import random
import zlib


import os
import io

import detectron2
from utils import IOProcessor

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# import some common libraries
import numpy as np
import cv2
import torch
import jsonlines

NUM_OBJECTS = 10

from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs

FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features',"title"]

MIN_BOXES = 36
MAX_BOXES = 36

parser = argparse.ArgumentParser()
parser.add_argument('--split', default='valxl', help='train2014, val2014')
parser.add_argument('--batchsize', default=4, type=int, help='batch_size')
parser.add_argument('--model', default='res5', type=str, help='options: "res4", "res5"; features come from)')
parser.add_argument('--weight', default='vg', type=str,
        help='option: mask, obj, vg. mask:mask_rcnn on COCO, obj: faster_rcnn on COCO, vg: faster_rcnn on Visual Genome')
parser.add_argument("--start", type=int)
parser.add_argument("--end", type=int)
args = parser.parse_args()


def read_json(file):
    f=open(file,"r",encoding="utf-8").read()
    return json.loads(f)

def write_json(file,data):
    f=open(file,"w",encoding="utf-8")
    json.dump(data,f,indent=2,ensure_ascii=False)
    return

def read_jsonline(file):
    file=open(file,"r",encoding="utf-8")
    data=[json.loads(line) for line in file.readlines()]
    return  data

def write_jsonline(file,data):
    f=jsonlines.open(file,"w")
    for each in data:
        jsonlines.Writer.write(f,each)
    return


def open_tsv(fname, folder):
    print("Opening %s Data File..." % fname)
    df = pd.read_csv(fname, sep='\t', names=["caption", "filename"], usecols=range(0, 2))
    df['folder'] = folder
    print("Processing", len(df), " Images:")
    return df

def write_to_tsv(output_path: str, file_columns: list, data: list):
    csv.register_dialect('tsv_dialect', delimiter='\t', quoting=csv.QUOTE_ALL)
    with open(output_path, "w", newline="") as wf:
        writer = csv.DictWriter(wf, fieldnames=file_columns, dialect='tsv_dialect')
        writer.writerows(data)
    csv.unregister_dialect('tsv_dialect')

def read_from_tsv(file_path: str, column_names: list) -> list:
    csv.register_dialect('tsv_dialect', delimiter='\t', quoting=csv.QUOTE_ALL)
    with open(file_path, "r") as wf:
        reader = csv.DictReader(wf, fieldnames=column_names, dialect='tsv_dialect')
        datas = []
        for row in reader:
            data = dict(row)
            datas.append(data)
    csv.unregister_dialect('tsv_dialect')
    return datas


def _file_name(row):
    return "%s/%s" % (row['folder'], (zlib.crc32(row['url'].encode('utf-8')) & 0xffffffff))


def load_image_ids(dir_path,image_type="danpin"):
    file_list=os.listdir(dir_path)
    image_file_ids=[]
    for each in file_list:
        file_path="{}/{}".format(dir_path,each)
        image_id=each.split(".")[0]
        image_file_ids.append((file_path,image_id,image_type))
    return image_file_ids

from torchvision.ops import nms
from detectron2.structures import Boxes, Instances
def fast_rcnn_inference_single_image(
        boxes, scores, image_shape, score_thresh, nms_thresh, topk_per_image
):
    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # Select max scores
    max_scores, max_classes = scores.max(1)  # R x C --> R
    num_objs = boxes.size(0)
    boxes = boxes.view(-1, 4)
    idxs = torch.arange(num_objs).cuda() * num_bbox_reg_classes + max_classes
    max_boxes = boxes[idxs]  # Select max boxes according to the max scores.

    # Apply NMS
    keep = nms(max_boxes, max_scores, nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores = max_boxes[keep], max_scores[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = max_classes[keep]

    return result, keep

def get_predictor():
    cfg = get_cfg()  # Renew the cfg file
    cfg.merge_from_file("../../configs/VG-Detection/faster_rcnn_R_101_C4_caffemaxpool.yaml")
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    cfg.INPUT.MIN_SIZE_TEST = 600
    cfg.INPUT.MAX_SIZE_TEST = 1000
    cfg.MODEL.RPN.NMS_THRESH = 0.7
    # Find a model from detectron2's model zoo. You can either use the https://dl.fbaipublicfiles.... url, or use the following shorthand
    # cfg.MODEL.WEIGHTS = "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe.pkl"
    # cfg.MODEL.WEIGHTS = "../../models/faster_rcnn_from_caffe.pkl"
    cfg.MODEL.WEIGHTS = "../../models/faster_rcnn_from_caffe_attr.pkl"
    # faster_rcnn_from_caffe_attr
    predictor = DefaultPredictor(cfg)
    print("predictor: ",predictor)

    return predictor


def doit(detector, raw_images):
    with torch.no_grad():
        # Preprocessing
        inputs = []
        for raw_image in raw_images:
            try:
                image = detector.transform_gen.get_transform(raw_image).apply_image(raw_image)
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                inputs.append({"image": image, "height": raw_image.shape[0], "width": raw_image.shape[1]})
            except Exception as e:
                print("continue ",e)
                continue
        images = detector.model.preprocess_image(inputs)

        # Run Backbone Res1-Res4
        features = detector.model.backbone(images.tensor)

        # Generate proposals with RPN
        proposals, _ = detector.model.proposal_generator(images, features, None)

        # Run RoI head for each proposal (RoI Pooling + Res5)
        proposal_boxes = [x.proposal_boxes for x in proposals]
        features = [features[f] for f in detector.model.roi_heads.in_features]
        box_features = detector.model.roi_heads._shared_roi_transform(
            features, proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # (sum_proposals, 2048), pooled to 1x1

        # Predict classes and boxes for each proposal.
        pred_class_logits, pred_proposal_deltas = detector.model.roi_heads.box_predictor(feature_pooled)
        rcnn_outputs = FastRCNNOutputs(
            detector.model.roi_heads.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            detector.model.roi_heads.smooth_l1_beta,
        )

        # Fixed-number NMS
        instances_list, ids_list = [], []
        probs_list = rcnn_outputs.predict_probs()
        boxes_list = rcnn_outputs.predict_boxes()
        for probs, boxes, image_size in zip(probs_list, boxes_list, images.image_sizes):
            for nms_thresh in np.arange(0.3, 1.0, 0.1):
                instances, ids = fast_rcnn_inference_single_image(
                    boxes, probs, image_size,
                    score_thresh=0.2, nms_thresh=nms_thresh, topk_per_image=MAX_BOXES
                )
                if len(ids) >= MIN_BOXES:
                    break
            instances_list.append(instances)
            ids_list.append(ids)

        # Post processing for features
        features_list = feature_pooled.split(
            rcnn_outputs.num_preds_per_image)  # (sum_proposals, 2048) --> [(p1, 2048), (p2, 2048), ..., (pn, 2048)]
        roi_features_list = []
        for ids, features in zip(ids_list, features_list):
            roi_features_list.append(features[ids].detach())

        # Post processing for bounding boxes (rescale to raw_image)
        raw_instances_list = []
        for instances, input_per_image, image_size in zip(
                instances_list, inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            raw_instances = detector_postprocess(instances, height, width)
            raw_instances_list.append(raw_instances)

        return raw_instances_list, roi_features_list

def dump_features(writer, detector, img_ids,image_prefix,gallery_unit_id_label):
    img_paths=["{}/{}.jpg".format(image_prefix,each) for each in img_ids]
    imgs = [cv2.imread(img_path) for img_path in img_paths]
    try:
        instances_list, features_list = doit(detector, imgs)

        for img, img_id, instances, features in zip(imgs, img_ids, instances_list, features_list):
            if type(img)==type(None) or features==None:
                continue

            instances = instances.to('cpu')
            features = features.to('cpu')

            num_objects = len(instances)

            item = {
                "image_id": img_id,
                "image_h": img.shape[0],
                "image_w": img.shape[1],
                "num_boxes": num_objects,
                "boxes": base64.b64encode(instances.pred_boxes.tensor.numpy()).decode(),  # float32
                "features": base64.b64encode(features.numpy()).decode(),  # float32
                "title":gallery_unit_id_label[img_id]["title"]
            }

            writer.writerow(item)
    except Exception as e:
        print("image: ",img_paths)
        print(e)


from tqdm import tqdm
def extract_feat(SP_train_id_list,image_prefix,outfile, detector,gallery_unit_id_label):
    # Check existing images in tsv file.
    wanted_ids = set([image_id for image_id in SP_train_id_list])
    found_ids = set()
    if os.path.exists(outfile):
        with open(outfile, 'r') as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter='\t', fieldnames=FIELDNAMES)
            for item in reader:
                found_ids.add(item['image_id'])
    print(len(found_ids))
    missing_ids = wanted_ids - found_ids
    missing_ids=list(missing_ids)

    with open(outfile, 'a') as tsvfile:
        writer = csv.DictWriter(tsvfile, delimiter='\t', fieldnames=FIELDNAMES)
        for start in tqdm(range(0, len(missing_ids), args.batchsize)):
            missing_ids_trunk = missing_ids[start: start + args.batchsize]
            dump_features(writer, detector, missing_ids_trunk,image_prefix,gallery_unit_id_label)


import argparse
if __name__ == '__main__':
    print()
    xl_io=IOProcessor()
    predictor = get_predictor()

    image_prefix="/data1/xl/images" # image root
    data_root="{your_data_root}"
    tsv_root="{your_tsv_root}"

    data_id_label=xl_io.read_json("{}/train_id_info.json".format(data_root))
    data_id_list=list(data_id_label.keys())

    extract_feat(data_id_list,image_prefix,"{}/train.tsv".format(tsv_root),predictor,data_id_label)

    # 23158





































