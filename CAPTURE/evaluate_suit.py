#!/usr/bin/env python
from __future__ import division
import sys
import os
import io
import os.path
import numpy as np
import json
import shutil
import argparse

# input_dir = sys.argv[1]
# output_dir = "/data1/xl/product_retrieval/evaluate"
#
# submit_dir = "/data1/xl/product_retrieval/evaluate"
# truth_dir = "/data1/xl/product_retrieval/all_data"


def parse_args():
    parser=argparse.ArgumentParser()

    parser.add_argument("--output_metric_dir",type=str)
    parser.add_argument("--retrieval_result_dir",type=str)

    parser.add_argument("--GT_dir",type=str)

    return parser.parse_args()


#
# if not os.path.isdir(submit_dir):
#     print ("%s doesn't exist" % submit_dir)
#
# if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

def read_json(file):
    f=io.open(file,"r",encoding="utf-8").read()
    f=f.encode("utf-8")
    return json.loads(f)

def write_json(file,data):
    f=open(file,"w")
    json.dump(data,f,indent=2,ensure_ascii=False)
    return


def compute_p(rank_list,pos_set,topk):
    intersect_size = 0
    for i in range(topk):
        if rank_list[i] in pos_set:
            intersect_size+=1

    p=float(intersect_size/topk)

    return p

def compute_ap(rank_list,pos_set,topk):
    '''
        rank_list:
        pos_list:
        rank_list=["a","d","b","c"]
        pos_set=["b","c"]
        ap=compute_ap(rank_list,pos_set)
        print("ap: ",ap)
    '''
    intersect_size=0
    ap=0

    for i in range(topk):
        if rank_list[i] in pos_set:
            intersect_size += 1
            precision = intersect_size / (i+1)
            ap+=precision
    if intersect_size==0:
        return 0
    ap/=intersect_size

    return ap

def compute_HitRate(rank_label_set,query_label_set):
    return len(rank_label_set.intersection(query_label_set))/len(query_label_set)


def compute_ar(rank_list,GT_labels,gallery_label_ids,gallery_id_label,N):
    rank_list=rank_list[:N]

    GT_label_set=list(set(GT_labels))
    label_expect_count={}
    for label in GT_labels:
        if label not in label_expect_count:
            label_expect_count[label]=int(N/len(GT_labels))
        else:
            label_expect_count[label]+=int(N/len(GT_labels))

    for label in label_expect_count:
        label_expect_count[label]=min(label_expect_count[label],len(gallery_label_ids[label]))


    label_retrieval_count={}
    for id in rank_list:
        label=gallery_id_label[id]["label"][0]
        if label in label_expect_count:
            if label not in label_retrieval_count:
                label_retrieval_count[label]=1
            else:
                label_retrieval_count[label]+=1

    for label in label_retrieval_count:
        label_retrieval_count[label]=min(label_retrieval_count[label],label_expect_count[label])

    label_ap=0


    for label in label_retrieval_count:
        label_ap+=label_retrieval_count[label]/label_expect_count[label]

    label_ap/=len(GT_label_set)
    return label_ap


def main():
    args = parse_args()

    if not os.path.exists(args.output_metric_dir):
        os.makedirs(args.output_metric_dir)


    gallery_unit_id_label_txt=open("{}/product1m_gallery_ossurl_v2.txt".format(args.GT_dir)).readlines()
    test_query_suit_id_label_txt = open("{}/product1m_test_ossurl_v2.txt".format(args.GT_dir)).readlines()
    dev_query_suit_id_label_txt=open("{}/product1m_dev_ossurl_v2.txt".format(args.GT_dir)).readlines()

    gallery_unit_id_label={}
    for line in gallery_unit_id_label_txt:
        line=line.strip()
        line_split=line.split("#####")
        item_id=line_split[0]
        label_list=line_split[4].split("#;#")
        gallery_unit_id_label[item_id]={
            "label":label_list
        }

    test_query_suit_id_label={}
    for line in test_query_suit_id_label_txt:
        line=line.strip()
        line_split=line.split("#####")
        item_id=line_split[0]
        label_list=line_split[4].split("#;#")
        test_query_suit_id_label[item_id]={
            "label":label_list
        }
    for line in dev_query_suit_id_label_txt:
        line=line.strip()
        line_split=line.split("#####")
        item_id=line_split[0]
        label_list=line_split[4].split("#;#")
        test_query_suit_id_label[item_id]={
            "label":label_list
        }


    gallery_unit_label_id={}
    for item_id,info in gallery_unit_id_label.items():
        label=info["label"][0]
        if label not in gallery_unit_label_id:
            gallery_unit_label_id[label]=[item_id]
        else:
            gallery_unit_label_id[label]+=[item_id]


    results={}

    retrieval_results=open("{}/retrieval_id_list.txt"
                           .format(args.retrieval_result_dir),"r").readlines()

    topk_list=[10,50,100]
    for topk in topk_list:
        topk_temp=topk
        mAP=0
        mP=0
        mAR=0
        cnt=0
        for index,each in enumerate(retrieval_results):
            each=each.strip()
            each_split=each.split(",")
            query_id=each_split[0]
            rank_id_list=each_split[1:]
            pos_set=[]

            try:
                cnt+=1
                query_suit_labels=test_query_suit_id_label[query_id]["label"]
                for label in query_suit_labels:
                    pos_set+=gallery_unit_label_id[label]

                topk = min(topk_temp, len(pos_set),len(rank_id_list))


                ap=compute_ap(rank_id_list,pos_set,topk)

                p=compute_p(rank_id_list,pos_set,topk)

                mAP+=ap
                mP+=p


                Ar=compute_ar(rank_list=rank_id_list,
                               GT_labels=query_suit_labels,
                               gallery_label_ids=gallery_unit_label_id,
                               gallery_id_label=gallery_unit_id_label,
                               N=topk_temp)
                mAR+=Ar

            except:
                print(query_id)


        mAP/=cnt
        mP/=cnt
        mAR/=cnt


        print("topk: {}  mAP: {} ".format(topk_temp,mAP))

        results["top{}".format(topk_temp)]={
            "mAP": mAP,
            "Prec":mP,
            "mAR":mAR
        }


    write_json("{}/metric_results.json".format(args.output_metric_dir),results)

    return

if __name__ == '__main__':
    main()






