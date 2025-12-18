import os
import re
import json
import argparse
import string
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import pandas as pd
from functools import partial


def clean_answer(data):
    data = data.lower()
    data = re.sub('[ ]+$' ,'', data)
    data = re.sub('^[ ]+' ,'', data)
    data = re.sub(' {2,}', ' ', data)

    data = re.sub('\.[ ]{2,}', '. ', data)
    data = re.sub('[^a-zA-Z0-9,\'\s\-:]+', '', data)
    data = re.sub('ç' ,'c', data)
    data = re.sub('’' ,'\'', data)
    data = re.sub(r'\bletf\b' ,'left', data)
    data = re.sub(r'\blet\b' ,'left', data)
    data = re.sub(r'\btehre\b' ,'there', data)
    data = re.sub(r'\brigth\b' ,'right', data)
    data = re.sub(r'\brght\b' ,'right', data)
    data = re.sub(r'\bbehine\b', 'behind', data)
    data = re.sub(r'\btv\b' ,'TV', data)
    data = re.sub(r'\bchai\b' ,'chair', data)
    data = re.sub(r'\bwasing\b' ,'washing', data)
    data = re.sub(r'\bwaslked\b' ,'walked', data)
    data = re.sub(r'\boclock\b' ,'o\'clock', data)
    data = re.sub(r'\bo\'[ ]+clock\b' ,'o\'clock', data)

    # digit to word, only for answer
    data = re.sub(r'\b0\b', 'zero', data)
    data = re.sub(r'\bnone\b', 'zero', data)
    data = re.sub(r'\b1\b', 'one', data)
    data = re.sub(r'\b2\b', 'two', data)
    data = re.sub(r'\b3\b', 'three', data)
    data = re.sub(r'\b4\b', 'four', data)
    data = re.sub(r'\b5\b', 'five', data)
    data = re.sub(r'\b6\b', 'six', data)
    data = re.sub(r'\b7\b', 'seven', data)
    data = re.sub(r'\b8\b', 'eight', data)
    data = re.sub(r'\b9\b', 'nine', data)
    data = re.sub(r'\b10\b', 'ten', data)
    data = re.sub(r'\b11\b', 'eleven', data)
    data = re.sub(r'\b12\b', 'twelve', data)
    data = re.sub(r'\b13\b', 'thirteen', data)
    data = re.sub(r'\b14\b', 'fourteen', data)
    data = re.sub(r'\b15\b', 'fifteen', data)
    data = re.sub(r'\b16\b', 'sixteen', data)
    data = re.sub(r'\b17\b', 'seventeen', data)
    data = re.sub(r'\b18\b', 'eighteen', data)
    data = re.sub(r'\b19\b', 'nineteen', data)
    data = re.sub(r'\b20\b', 'twenty', data)
    data = re.sub(r'\b23\b', 'twenty-three', data)

    # misc
    # no1, mat2, etc
    data = re.sub(r'\b([a-zA-Z]+)([0-9])\b' ,r'\g<1>', data)
    data = re.sub(r'\ba\b ([a-zA-Z]+)' ,r'\g<1>', data)
    data = re.sub(r'\ban\b ([a-zA-Z]+)' ,r'\g<1>', data)
    data = re.sub(r'\bthe\b ([a-zA-Z]+)' ,r'\g<1>', data)

    data = re.sub(r'\bbackwards\b', 'backward', data)

    return data


MCA_QUESTION_TYPES = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
]
NA_QUESTION_TYPES = [
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
]

METRICS_FOR_MCA = {
    "accuracy": "exact_match",
}

METRICS_FOR_NA = {
    "MRA:.5:.95:.05": "partial(mean_relative_accuracy, start=.5, end=.95, interval=.05)",
}

def fuzzy_matching(pred):
    return pred.split(' ')[0].rstrip('.').strip()

def exact_match(pred, target):
    return 1. if pred.lower() == target.lower() else 0.

def abs_dist_norm(pred, target):
    return abs(pred - target) / target

def mean_relative_accuracy(pred, target, start, end, interval):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()

WORST_CASE_FOR_METRICS = {
    "accuracy": 0.,
    "MRA:.5:.95:.05": 0.,
}

def to_float(pred):
    try:
        pred = float(pred)
    except BaseException as e:
        pred = None
    return pred

def vsibench_process_results(doc):
    if doc['question_type'] in MCA_QUESTION_TYPES:
        for key, value in METRICS_FOR_MCA.items():
            doc[key] = eval(value)(fuzzy_matching(doc['pred_response']), doc['gt_response'])
    elif doc['question_type'] in NA_QUESTION_TYPES:
        for key, value in METRICS_FOR_NA.items():
            try:
                doc[key] = eval(value)(to_float(fuzzy_matching(doc['pred_response'])), to_float(doc['gt_response']))
            except TypeError:
                doc[key] = WORST_CASE_FOR_METRICS[key]
    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")

    return doc

# refer to LEO: embodied-generalist
# https://github.com/embodied-generalist/embodied-generalist/blob/477dc44b8b18dbfbe6823c307436d896ec8b062e/evaluator/scanqa_eval.py#L41-L50
def answer_match(pred, gts):
    # return EM and refined EM
    if pred in gts:
        return 1, 1
    for gt in gts:
        if ''.join(pred.split()) in ''.join(gt.split()) or ''.join(gt.split()) in ''.join(pred.split()):
            return 0, 1
    return 0, 0


def main(args):
    with open(args.input_file) as f:
        data = json.load(f)

    import sys;sys.path.append(".")
    from eval.vlm.caption_eval.bleu.bleu import Bleu
    from eval.vlm.caption_eval.rouge.rouge import Rouge
    from eval.vlm.caption_eval.meteor.meteor import Meteor
    from eval.vlm.caption_eval.cider.cider import Cider

    if args.dataset == "scan2cap":
        # For Scan2cap
        cider = Cider()
        bleu = Bleu()
        meteor = Meteor()
        rouge = Rouge()

        res, gts = {}, {}
        for item in data:
            res[item['sample_id']] = ['sos ' + item['pred_response'].replace('.', ' . ').replace(',', ' , ').lower() + ' eos' ]
            gts[item['sample_id']] = ['sos ' + it.replace('.', ' . ').replace(',', ' , ').lower() + ' eos' for it in item['gt_response']]

        cider_score = cider.compute_score(gts, res)
        bleu_score = bleu.compute_score(gts, res)
        meteor_score = meteor.compute_score(gts, res)
        rouge_score = rouge.compute_score(gts, res)

        print(f"CIDER: {cider_score[0]*100}")
        print(f"BLEU: {bleu_score[0][-1]*100}")
        print(f"METEOR: {meteor_score[0]*100}")
        print(f"Rouge: {rouge_score[0]*100}")
    elif args.dataset == "sqa3d":
        # For SQA3D
        from collections import defaultdict
        correct_per_type = defaultdict(list)
        correct_per_type_refine = defaultdict(list)

        for item in data:
            pred = clean_answer(item['pred_response'])
            gt = clean_answer(item['gt_response'])

            em, em_refine = answer_match(pred, [gt])

            correct_per_type["all"].append(em)
            correct_per_type[item['question_type']].append(em)

            correct_per_type_refine["all"].append(em_refine)
            correct_per_type_refine[item['question_type']].append(em_refine)

        for k in correct_per_type:
            print(f"EM-{k}: {np.mean(correct_per_type[k]) * 100}")
        
        print()
        
        for k in correct_per_type:
            print(f"EM-R-{k}: {np.mean(correct_per_type_refine[k]) * 100}")
    elif args.dataset == "scanqa":
        # For ScanQA
        with open("/mnt/workspace/cv_multimodal/aigc/dataset/video_3d_llm_data/data/processed/scanqa_val_llava_style.json") as f:
            raw_data = json.load(f)
            idx2labels = {}
            for item in raw_data:
                idx2labels[item['id']] = item['metadata']['answers']

        cider = Cider()
        bleu = Bleu()
        meteor = Meteor()
        rouge = Rouge()

        n_correct = 0
        res, gts = {}, {}
        for item in data:
            item["sample_id"] = "_".join(item["sample_id"].split("_")[:-1] + ['0'])
            res[item['sample_id']] = [item['pred_response'].rstrip(".")]
            gts[item['sample_id']] = idx2labels[item['sample_id']]

            if item['pred_response'] in idx2labels[item['sample_id']]:
                n_correct += 1

        cider_score = cider.compute_score(gts, res)
        bleu_score = bleu.compute_score(gts, res)
        meteor_score = meteor.compute_score(gts, res)
        rouge_score = rouge.compute_score(gts, res)

        print(f"count: {len(gts)}")
        print(f"CIDER: {cider_score[0]*100}")
        print(f"BLEU: {bleu_score[0][0]*100}, {bleu_score[0][1]*100}, {bleu_score[0][2]*100}, {bleu_score[0][3]*100}")
        print(f"METEOR: {meteor_score[0]*100}")
        print(f"Rouge: {rouge_score[0]*100}")
        print(f"EM: {n_correct*100 / len(data)}")
    elif args.dataset == "vsibench":
        # For VSI-Bench
        results = []
        for idx, item in enumerate(data):
            results.append(vsibench_process_results(item))

        results = pd.DataFrame(results)

        output = {}

        for question_type, question_type_indexes in results.groupby('question_type').groups.items():
            per_question_type = results.iloc[question_type_indexes]
            
            if question_type in MCA_QUESTION_TYPES:
                for metric in METRICS_FOR_MCA.keys():
                    output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
            elif question_type in NA_QUESTION_TYPES:
                for metric in METRICS_FOR_NA.keys():
                    if metric == 'success_rate':
                        output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
                    else:
                        output[f"{question_type}_{metric}"] = per_question_type[metric].mean()

            else:
                raise ValueError(f"Unknown question type: {question_type}")
        
        output['object_rel_direction_accuracy'] = sum([
            output.pop('object_rel_direction_easy_accuracy'),
            output.pop('object_rel_direction_medium_accuracy'),
            output.pop('object_rel_direction_hard_accuracy'),
        ]) / 3.
        
        output['overall'] = sum([_ for _ in output.values()]) / len(output)

        for key, value in output.items():
            print(f"{key}:", value * 100)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='sqa3d')
    parser.add_argument('--input-file', type=str, default='results/sqa3d/test.jsonl')
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("-n", type=int, default=-1)
    args = parser.parse_args()

    main(args)
