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
# import faulthandler
# faulthandler.enable()

import datetime
import argparse
import itertools
import json
import os
import random
from typing import Optional
import numpy as np

import torch
from eval.vlm.utils import load_model_and_tokenizer, build_transform, process_conversation
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from decord import VideoReader, cpu

# os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'

ds_collections = {
    'vsibench': {
        'test': './dataset/eval/VSI-Bench',
        'metric': None,
        'max_new_tokens': 16,
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


def read_video_images(video_file):
    # read video images from the source
    if not os.path.exists(video_file):
        print(f"File not exist: {video_file}")
        raise FileNotFoundError
    
    def get_frame_indices(total_frames, fps=1):
        video_length = total_frames / fps
        interval = 2
        num_frames_to_sample = round(video_length / interval)
        video_min_frames = 8
        video_max_frames = 32
        target_frames = min(
            max(num_frames_to_sample, video_min_frames), video_max_frames
        )
        frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frame_idx = np.unique(frame_idx)
        return frame_idx
    
    # check whether video_file is a directory
    if os.path.isdir(video_file):
        frame_files = [os.path.join(video_file, f) for f in os.listdir(video_file) if os.path.isfile(os.path.join(video_file, f))]
        frame_files.sort()
        frame_idx = get_frame_indices(len(frame_files), 1)
        images = [frame_files[i] for i in frame_idx]
        images = [Image.open(frame).convert("RGB") for frame in images]
    elif any([video_file.endswith(ext) for ext in [".mp4", ".avi", ".mov"]]):
        vr = VideoReader(video_file, num_threads=4)
        total_frames = len(vr)
        avg_fps = vr.get_avg_fps()
        frame_idx = get_frame_indices(total_frames, avg_fps)
        video = vr.get_batch(frame_idx).asnumpy()
        
        images = [Image.fromarray(frame).convert("RGB") for frame in video]
    return images


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


class VQADataset(torch.utils.data.Dataset):

    def __init__(self, test):
        self.bench_path = test
        self.bench = load_dataset(self.bench_path)["test"]

    def __len__(self):
        return len(self.bench)

    def __getitem__(self, idx):
        item = self.bench.__getitem__(idx)

        dataset_name = item["dataset"]
        scene_id = item["scene_name"]
        scene_path = os.path.join(self.bench_path, dataset_name, scene_id) + ".mp4"
        
        images = read_video_images(scene_path)

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

        question = item["question"]
        if item['question_type'] in NA_QUESTION_TYPES:
            post_prompt = "Please answer the question using a single word or phrase."
            question = "<image>\n" + question + "\n" + post_prompt
        elif item['question_type'] in MCA_QUESTION_TYPES:
            options = "Options:\n" + "\n".join(item["options"])
            post_prompt =  "Answer with the option's letter from the given choices directly."
            question = "\n".join([question, options, post_prompt])
            question = "<image>" + question

        images, conversation = process_conversation(images, question)

        return {
            'dataset_name': dataset_name,
            'question_id': idx,
            'question_type': item["question_type"],
            'video_id': scene_id,
            'question': question,
            'image': images,
            'conversation': conversation,
            'answer': item["ground_truth"],
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
    random.seed(args.seed)

    for ds_name in args.datasets:
        dataset = VQADataset(
            test=ds_collections[ds_name]['test'],
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

        torch.cuda.empty_cache()
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
    parser.add_argument('--datasets', type=str, default='vsibench')
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
    model = model.to(torch.bfloat16)
    image_transform = build_transform()

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f'[test] total_params: {total_params}B')

    evaluate_chat_model()
