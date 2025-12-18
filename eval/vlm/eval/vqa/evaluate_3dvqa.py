# Copyright (c) 2023 OpenGVLab
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: MIT
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025-05-20.
#
# Original file was released under MIT, with the full license text
# available at https://github.com/OpenGVLab/InternVL/blob/main/LICENSE.
#
# This modified file is released under the same license.

import datetime
import argparse
import itertools
import json
import os
import random
import pickle
import subprocess
from typing import Optional
import numpy as np

import torch
from eval.vlm.utils import load_model_and_tokenizer, build_transform, process_conversation
from PIL import Image
from tqdm import tqdm

# os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'

ds_collections = {
    'scan2cap': {
        'test': './dataset/eval/3dscene/scan2cap_val_32frames.json',
        'metric': None,
        'max_new_tokens': 100,
    },
    'scanqa': {
        'test': './dataset/eval/3dscene/scanqa_val_llava_style.json',
        'metric': None,
        'max_new_tokens': 100,
    },
    'sqa3d': {
        'test': './dataset/eval/3dscene/sqa3d_test_llava_style.json',
        'metric': None,
        'max_new_tokens': 100,
    },
    '3ddet': {
        'test': './dataset/eval/3dscene/scannet_det_val_4frames.json',
        'metric': None,
        'max_new_tokens': 2000,
    },
    'scanrefer': {
        'test': './dataset/eval/3dscene/scanrefer_val_32frames.json',
        'metric': None,
        'max_new_tokens': 100,
    },
}


# https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py#L81
def relaxed_correctness(target: str,
                        prediction: str,
                        max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    Args:
      target: Target string.
      prediction: Predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str) -> Optional[float]:
        try:
            if text.endswith('%'):
                # Convert percentages to floats.
                return float(text.rstrip('%')) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float -
                              target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()


def evaluate_relaxed_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem['annotation'], str):
            elem['annotation'] = [elem['annotation']]
        score = max([
            relaxed_correctness(elem['answer'].strip(), ann)
            for ann in elem['annotation']
        ])
        scores.append(score)
    return sum(scores) / len(scores)


def evaluate_exact_match_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem['annotation'], str):
            elem['annotation'] = [elem['annotation']]
        score = max([
            (1.0 if
             (elem['answer'].strip().lower() == ann.strip().lower()) else 0.0)
            for ann in elem['annotation']
        ])
        scores.append(score)
    return sum(scores) / len(scores)


def collate_fn(batches):
    dataset_names = [_['dataset_name'] for _ in batches]
    question_ids = [_['question_id'] for _ in batches]
    question_types = [_['question_type'] for _ in batches]
    video_ids = [_['video_id'] for _ in batches]
    questions = [_['question'] for _ in batches]
    images = [_['image'] for _ in batches]
    conversations = [_['conversation'] for _ in batches]
    answers = [_['answer'] for _ in batches]

    return dataset_names, question_ids, question_types, video_ids, questions, images, conversations, answers


class VQADataset(torch.utils.data.Dataset):

    def __init__(self, test, prompt):
        with open(test, "r") as test_file:
            self.test = json.load(test_file)
        self.prompt = prompt

        self.scene = {}
        for split in ["train", "val", "test"]:
            with open(os.path.join("./dataset/eval/embodiedscan", f"embodiedscan_infos_{split}.pkl"), "rb") as f:
                data = pickle.load(f)["data_list"]
                for item in data:
                    # e.g., item["sample_idx"]: "scannet/scene0415_00"
                    if item["sample_idx"].startswith("scannet"):
                        self.scene[item["sample_idx"]] = item
        
        sampling_file = "./dataset/eval/3dscene/scannet_select_frames.json"
        self.mc_sampling_files = {}
        with open(sampling_file) as f:
            data = json.load(f)
            for dd in data:
                self.mc_sampling_files[dd['video_id']] = dd

    def __len__(self):
        return len(self.test)

    def __getitem__(self, idx):
        data = self.test[idx]
        try:
            dataset_name = data["metadata"]["dataset"].lower()
        except:
            dataset_name = "3ddet"

        answers = None
        if dataset_name in ["scanqa", "sqa3d"]:
            video_id, question, question_id, question_type = data['video'], data["conversations"][0]["value"], data['id'], data['metadata']['question_type']
            # For ScanQA, SQA3D
            answers = data["conversations"][1]["value"]
        elif dataset_name == "scan2cap":
            frame_files, question, question_type = data['images'], data["conversations"][0]["value"], data['metadata']['question_type']
            question = "<image>" + question.replace("<image>", "")
            video_id = "scannet/" + frame_files[0].split("/")[-2]
            question_id = idx
            # For Scan2cap
            answers = {"answer": data.get("annotations", [data["conversations"][1]["value"]]), "iou": data["iou"]}
        elif dataset_name == "3ddet":
            # For 3D Det
            frame_files, question = data['images'], data["conversations"][0]["value"]
            video_id = "scannet/" + frame_files[0].split("/")[-2]
            question_id = idx
            question_type = "detection"
            answers = data["conversations"][1]["value"]
        elif dataset_name == "scanrefer":
            # For 3D Grounding
            frame_files, question, question_type = data['images'], data["conversations"][0]["value"], data['metadata']['question_type']
            question = question.replace("Frame-0: <image>Frame-1: <image>Frame-2: <image>Frame-3: <image>Frame-4: <image>Frame-5: <image>Frame-6: <image>Frame-7: <image>Frame-8: <image>Frame-9: <image>Frame-10: <image>Frame-11: <image>Frame-12: <image>Frame-13: <image>Frame-14: <image>Frame-15: <image>Frame-16: <image>Frame-17: <image>Frame-18: <image>Frame-19: <image>Frame-20: <image>Frame-21: <image>Frame-22: <image>Frame-23: <image>Frame-24: <image>Frame-25: <image>Frame-26: <image>Frame-27: <image>Frame-28: <image>Frame-29: <image>Frame-30: <image>Frame-31: <image>\n", "<image>\n")
            video_id = "scannet/" + frame_files[0].split("/")[-2]
            question_id = idx
            answers = {"gt_bbox": data["gt_bbox"], "axis_align_matrix": data["axis_align_matrix"], "cam2global": data["cam2global"]}

        if dataset_name in ["scanqa", "sqa3d"]:
            # NOTE: Hard CODE: Determine the indices for uniformly sampling 32 frames or set upbound to 32 in mc sampling
            # num_frames_to_sample = 16
            num_frames_to_sample = 32

            ### Uniform sample images, from: Video3DLLM/llava/video_utils.py lines 170-194
            # since the color images have the suffix .jpg
            # frame_files = [os.path.join(video_file, f) for f in os.listdir(video_file) if os.path.isfile(os.path.join(video_file, f)) and os.path.join(video_file, f).endswith(".jpg")]
            # frame_files.sort()  # Ensure the frames are sorted if they are named sequentially

            meta_info = self.scene[video_id]
            frame_files = [img["img_path"] for img in meta_info["images"]]

            # For scannet, the RGB camera data is temporally synchronized with the depth sensor via hardware, providing synchronized depth and color capture at 30Hz
            # We follow embodiedscan by sampling one out of every ten images.
            # avg_fps = 3
            
            total_frames = len(frame_files)
            sampled_indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)

            frame_files = [frame_files[i] for i in sampled_indices]
            ### Sample images end.

            # # ### MC sample images,
            # mc_files = self.mc_sampling_files[video_id]
            # frame_files = mc_files['frame_files'][:num_frames_to_sample]
            # voxel_nums = mc_files['voxel_nums'][:num_frames_to_sample]

            # ratio = 1.0
            # if 'ratio95' in self.frame_sampling_strategy:
            #     ratio = 0.95
            # elif 'ratio90' in self.frame_sampling_strategy:
            #     ratio = 0.9

            # if ratio != 1.0:
            #     num_all_voxels = mc_files['num_all_voxels']
            #     out = []
            #     cc = 0
            #     for frame_file, voxel_num in zip(frame_files, voxel_nums):
            #         out.append(frame_file)
            #         cc += voxel_num
            #         if cc >= num_all_voxels * ratio:
            #             break
            #     frame_files = out
            
            # frame_files.sort(key=lambda file: int(file.split('/')[-1].split('.')[0]))
            # # ### Sample images end.

        ### Read images.
        images = []
        for frame_path in frame_files:
            image = Image.open(os.path.join("./dataset/eval/", frame_path)).convert('RGB')
            images.append(image)

        # resize them
        H, W = images[0].size
        crop_size = 480 # NOTE: Hard CODE large: 480
        new_height = crop_size
        new_width = int(W * (crop_size / H))
        images = [frame.resize((new_width, new_height)) for frame in images]
        # # Calculate the position and perform the center crop NOTE: not crop in BAGEL
        # left = (new_width - crop_size) // 2
        # right = left + crop_size
        # top = (new_height - crop_size) // 2
        # bottom = top + crop_size
        # images = [frame.crop((left, top, right, bottom)) for frame in images]
        ### Read images end.

        if len(self.prompt) != 0:
            question = self.prompt + "\n" + question.replace("<image>", "")

        images, conversation = process_conversation(images, question)

        return {
            'dataset_name': dataset_name,
            'question_id': question_id,
            'question_type': question_type,
            'video_id': video_id,
            'question': question,
            'image': images,
            'conversation': conversation,
            'answer': answers,
        }


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def post_process(response):
    response = response.strip().split('.')[0].split(
        ',')[0].split('!')[0].lower()
    if 'is ' in response:
        response = response.split('is ')[1]
    if 'are ' in response:
        response = response.split('are ')[1]
    if 'a ' in response:
        response = response.split('a ')[1]
    if 'an ' in response:
        response = response.split('an ')[1]
    if 'the ' in response:
        response = response.split('the ')[1]
    if ' of' in response:
        response = response.split(' of')[0]
    response = response.strip()
    return response


def evaluate_chat_model():
    # base_prompt = 'Answer the question using a single word or phrase.'
    # vizwiz_prompt = "When the provided information is insufficient, respond with 'Unanswerable'. "
    # infovqa_prompt = 'Answer the question using a single word or phrase.'
    # spatial_prompt = f"The video captures 3D spatial information of a scene. Please focus on the spatial relationships in the video and answer the following questions."
    # ai2d_prompt = ''
    random.seed(args.seed)
    summaries = []

    for ds_name in args.datasets:
        input_prompt = f'<image>'

        dataset = VQADataset(
            test=ds_collections[ds_name]['test'],
            prompt=input_prompt,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=InferenceSampler(len(dataset)),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

        outputs = []
        for _, (dataset_names, question_ids, question_types, video_ids, questions, images, conversations, answers) in tqdm(enumerate(dataloader)):
            pred = model.chat(
                tokenizer, 
                new_token_ids,
                image_transform,
                images=images[0], # batch=1
                prompt=conversations[0], # batch=1
                max_length=ds_collections[ds_name]['max_new_tokens'],
            )
            preds = [pred]

            for question, question_id, question_type, pred, answer, video_id in zip(questions, question_ids, question_types, preds, answers, video_ids):
                if ds_name in ['scanqa', 'sqa3d', '3ddet', 'scanrefer']:
                    outputs.append({
                        "dataset": ds_name,
                        "sample_id": question_id,
                        "prompt": question,
                        "pred_response": pred,
                        "gt_response": answer,
                        "model_id": "BAGEL-7B",
                        "question_type": question_type,
                        "scene": video_id,
                    })
                elif ds_name == "scan2cap":
                    if answer["iou"] > 0.5:
                        outputs.append({
                            "dataset": ds_name,
                            "sample_id": question_id,
                            "prompt": question,
                            "pred_response": pred,
                            "gt_response": answer["answer"],
                            "model_id": "BAGEL-7B",
                            "question_type": question_type,
                            "scene": video_id,
                        })
                else:
                    raise NotImplementedError

        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_outputs = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

        merged_outputs = [json.loads(_) for _ in merged_outputs]
        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

        if torch.distributed.get_rank() == 0:
            print(f'Evaluating {ds_name} ...')
            results_file = f'{ds_name}.json'
            results_file = os.path.join(args.out_dir, results_file)
            json.dump(merged_outputs, open(results_file, 'w'))
            print('Results saved to {}'.format(results_file))

        torch.distributed.barrier()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='scan2cap')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model-path', type=str, default='hf/BAGEL-7B-MoT/')
    parser.add_argument('--safetensor-path', type=str, default='')
    parser.add_argument('--few-shot', type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)
    assert args.batch_size == 1, 'Only batch size 1 is supported'

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
        timeout=datetime.timedelta(seconds=12000)
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    model, tokenizer, new_token_ids = load_model_and_tokenizer(args, args.safetensor_path)
    image_transform = build_transform()

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f'[test] total_params: {total_params}B')

    evaluate_chat_model()
