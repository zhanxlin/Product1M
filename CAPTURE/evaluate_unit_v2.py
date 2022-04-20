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

    parser.add_argument("--GT_file",type=str)

    parser.add_argument("--t",action="store_true")
    parser.add_argument("--p",action="store_true")
    parser.add_argument("--i",action="store_true")
    parser.add_argument("--v",action="store_true")
    parser.add_argument("--a",action="store_true")

    parser.add_argument("--tp",action="store_true")
    parser.add_argument("--ti", action="store_true")
    parser.add_argument("--tv", action="store_true")
    parser.add_argument("--pi", action="store_true")
    parser.add_argument("--pv", action="store_true")
    parser.add_argument("--iv", action="store_true")
    parser.add_argument("--ta", action="store_true")
    parser.add_argument("--pa", action="store_true")
    parser.add_argument("--ia", action="store_true")
    parser.add_argument("--va", action="store_true")


    parser.add_argument("--tpi", action="store_true")
    parser.add_argument("--tpv", action="store_true")
    parser.add_argument("--tiv", action="store_true")
    parser.add_argument("--piv", action="store_true")

    parser.add_argument("--tpiv", action="store_true")
    parser.add_argument("--tpiva", action="store_true")
    parser.add_argument("--dense", action="store_true")

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



def main():
    args = parse_args()

    if not os.path.exists(args.output_metric_dir):
        os.makedirs(args.output_metric_dir)

    feature_type = []
    if args.t: feature_type.append("t")
    if args.p: feature_type.append("p")
    if args.i: feature_type.append("i")
    if args.v: feature_type.append("v")
    if args.a: feature_type.append("a")

    if args.tp: feature_type.append("tp")
    if args.ti: feature_type.append("ti")
    if args.tv: feature_type.append("tv")
    if args.pi: feature_type.append("pi")
    if args.pv: feature_type.append("pv")
    if args.iv: feature_type.append("iv")
    if args.ta: feature_type.append("ta")
    if args.pa: feature_type.append("pa")
    if args.ia: feature_type.append("ia")
    if args.va: feature_type.append("va")

    if args.tpi: feature_type.append("tpi")
    if args.tpv: feature_type.append("tpv")
    if args.tiv: feature_type.append("tiv")
    if args.piv: feature_type.append("piv")

    if args.tpiv: feature_type.append("tpiv")
    if args.tpiva: feature_type.append("tpiva")
    if args.dense: feature_type.append("dense")

    # gallery_unit_id_label_txt=open("{}/gallery_unit_id_label.txt".format(args.GT_dir)).readlines()
    # test_query_suit_id_label_txt = open("{}/test_query_suit_id_label.txt".format(args.GT_dir)).readlines()

    all_id_label_temp=open("{}".format(args.GT_file),"r",encoding='utf-8').read()
    all_id_label_temp=json.loads(all_id_label_temp)
    all_id_label={}


    for id,info in all_id_label_temp.items():
        all_id_label[id]={
            "title":info["title"],
            "label":[info["label"]]
        }

    # print("all_id_label: ",all_id_label)

    # gallery_unit_id_label={}
    # for line in gallery_unit_id_label_txt:
    #     line=line.strip()
    #     line_split=line.split("#####")
    #     item_id=line_split[0]
    #     label_list=line_split[1].split("#;#")
    #     gallery_unit_id_label[item_id]={
    #         "label":label_list
    #     }
    #
    # test_query_suit_id_label={}
    # for line in test_query_suit_id_label_txt:
    #     line=line.strip()
    #     line_split=line.split("#####")
    #     item_id=line_split[0]
    #     label_list=line_split[1].split("#;#")
    #     test_query_suit_id_label[item_id]={
    #         "label":label_list
    #     }


    all_label_id={}
    for item_id,info in all_id_label.items():
        label=info["label"][0]
        if label not in all_label_id:
            all_label_id[label]=[item_id]
        else:
            all_label_id[label]+=[item_id]


    results={}
    for each_feature_type in feature_type:
        results[each_feature_type]={}

        retrieval_results=open("{}/{}_feature_retrieval_id_list.txt"
                               .format(args.retrieval_result_dir,each_feature_type),"r").readlines()

        topk_list=[1,5,10]
        for topk in topk_list:
            topk_temp=topk
            mAP=0
            mHitRate=0
            mP=0
            cnt=0
            for index,each in enumerate(retrieval_results):
                each=each.strip()
                each_split=each.split(",")
                query_id=each_split[0]
                rank_id_list=each_split[1:]
                pos_set=[]

                # try:
                cnt+=1
                query_suit_labels=all_id_label[query_id]["label"]
                for label in query_suit_labels:
                    pos_set+=all_label_id[label]

                topk = min(topk_temp, len(pos_set),len(rank_id_list))

                # if topk<10:
                #     print()
                #     print("query_suit_labels: ",query_suit_labels)
                #     print("topk in: ",topk)
                #     print("pos set: ",len(pos_set))
                #     print("rank id list: ",len(rank_id_list))
                #     print()

                ap=compute_ap(rank_id_list,pos_set,topk)
                p = compute_p(rank_id_list, pos_set, topk)

                # print("ap: ",ap)
                # print()


                mAP+=ap
                mP += p

                    # # hit rate
                    # query_suit_label_set = set(gallery_unit_id_label[query_id]["label"])
                    # rank_label_set = set([gallery_unit_id_label[item_id]["label"][0] for item_id in rank_id_list[:topk]])

                    # hit_rate=compute_HitRate(rank_label_set,query_suit_label_set)
                    # mHitRate+=hit_rate

                # except Exception as e:
                #     print(e)
                    # print(query_id)

                    # continue
                # if index==100:
                #     break


            mAP/=cnt
            mHitRate/=cnt
            mP /= cnt


            # print("topk: ",topk)
            # print("topk_temp: ",topk_temp)
            print("topk: {}  mAP: {} ".format(topk_temp,mAP))

            results[each_feature_type]["top{}".format(topk_temp)]={
                "mAP": mAP*100,
                "mHitRate": mHitRate*100,
                "Prec": mP*100,
                "average": 100*(mAP + mHitRate + mP) / 3
            }

    write_json("{}/metric_results.json".format(args.output_metric_dir),results)

    return

if __name__ == '__main__':
    main()




















