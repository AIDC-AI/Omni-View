# Modified by AIDC-AI, 2025
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, 
# software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, 
# either express or implied. See the License for the specific language governing permissions and limitations under the License.

import os
import json
from tqdm import tqdm
import torch
import torch.multiprocessing as mp
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor

import cv2
cv2.setNumThreads(16)

import numpy as np
import oss2
from PIL import Image
import torchvision

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from moge.model.v1 import MoGeModel as MoGeV1Model
from moge.model.v2 import MoGeModel as MoGeV2Model


device = "cuda"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

# Load the model from huggingface hub (or load from local). 
moge1_model = MoGeV1Model.from_pretrained("Ruicheng/moge-vitl").to(device)
moge2_model = MoGeV2Model.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)

# Load and preprocess example images (replace with your own image paths)
scene_path = "/path/to/re10k/train/images/"
depth_path = "/path/to/re10k/train/depths/"

from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
skyseg_processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")
skyseg_model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large").to(device)
# 定义天空类别的ID 119
SKY_CLASS_ID = 119


@torch.no_grad()
def segment_sky_with_oneformer(image):
    # image = Image.open(image_path)
    inputs = skyseg_processor(images=image, task_inputs=["semantic"], return_tensors="pt").to(device)
    
    outputs = skyseg_model(**inputs)
    
    # 获取语义分割结果
    predicted_semantic_map = skyseg_processor.post_process_semantic_segmentation(outputs, \
        target_sizes=[image.size[::-1]])[0]

    # 提取天空区域
    sky_mask = (predicted_semantic_map == SKY_CLASS_ID).float()

    return sky_mask


def get_valid_depth(depths_vggt, depths_vggt_conf, depths_moge, inputs_rgb):
    moge_align_depth_list = []
    valid_mask_list = []
    all_valid_max_list = []

    sky_masks = []
    for input_rgb in inputs_rgb:
        # segmentation sky
        sky_masks.append(segment_sky_with_oneformer(input_rgb)) # 天空区域为True
    # erosion sky
    sky_masks = torch.stack(sky_masks)
    sky_masks = (- torch.nn.functional.max_pool2d(-sky_masks, 3, stride=1, padding=1)) > 0

    depths_vggt = depths_vggt.squeeze(1)
    depths_vggt_conf = depths_vggt_conf.squeeze(1)

    valid_masks = (                              # (H, W)
        torch.isfinite(depths_moge) &               
        (depths_moge > 0) &                      
        torch.isfinite(depths_vggt) &              
        (depths_vggt > 0) &
        ~sky_masks                 # 非天空区域
    )

    # Remove outliers 2/8分最合适
    outlier_quantiles = torch.tensor([0.2, 0.8], device=device)
    
    for depth_vggt, depth_vggt_conf, depth_moge, valid_masks in zip(depths_vggt, depths_vggt_conf, depths_moge, valid_masks):
        # depth_moge 无效部分 设置为 有效部分最大值的1.5倍   避免final_align_depth出现负数
        depth_moge[~valid_masks] = depth_moge[valid_masks].max() * 1

        source_inv_depth = 1.0 / depth_moge
        target_inv_depth = 1.0 / depth_vggt
        
        # print(f'倒数值:{source_inv_depth.min()}, {source_inv_depth.max()}')    # 0.03 ～ 2.2

        source_mask, target_mask = valid_masks, valid_masks

        source_data_low, source_data_high = torch.quantile(
            source_inv_depth[source_mask], outlier_quantiles
        )
        target_data_low, target_data_high = torch.quantile(
            target_inv_depth[target_mask], outlier_quantiles
        )
        source_mask = (source_inv_depth > source_data_low) & (
            source_inv_depth < source_data_high
        )
        target_mask = (target_inv_depth > target_data_low) & (
            target_inv_depth < target_data_high
        )
        
        mask = torch.logical_and(source_mask, target_mask)
        mask = torch.logical_and(mask, valid_masks)

        source_data = source_inv_depth[mask].view(-1, 1)
        target_data = target_inv_depth[mask].view(-1, 1)

        ones = torch.ones((source_data.shape[0], 1), device=device)
        source_data_h = torch.cat([source_data, ones], dim=1)
        transform_matrix = torch.linalg.lstsq(source_data_h, target_data).solution

        scale, bias = transform_matrix[0, 0], transform_matrix[1, 0]
        aligned_inv_depth = source_inv_depth * scale + bias
        
        valid_inv_depth = aligned_inv_depth > 0  # 创建新的有效掩码
        valid_masks = valid_masks & valid_inv_depth  # 合并到原有效掩码
        valid_mask_list.append(valid_masks)
                    
        final_align_depth = 1.0 / aligned_inv_depth
        moge_align_depth_list.append(final_align_depth)
        
        all_valid_max_list.append(final_align_depth[valid_masks].max().item())

    return moge_align_depth_list, valid_mask_list, all_valid_max_list


def get_metric_scale(metric_moge_depth, moge_depth, valid_mask):
    # 提取有效区域数据
    valid_metric = metric_moge_depth[valid_mask]
    valid_moge = moge_depth[valid_mask]
    
    # 分位数差计算
    metric_diff = torch.quantile(valid_metric, 0.8) - torch.quantile(valid_metric, 0.2)
    moge_diff = torch.quantile(valid_moge, 0.8) - torch.quantile(valid_moge, 0.2)
    return metric_diff / moge_diff


def align_metric_depth(moge1_depth_list, moge2_depth_list, valid_mask_list):
    
    metric_scales_list = [get_metric_scale(moge2_depth, moge1_depth, valid_mask) for (moge2_depth, moge1_depth, valid_mask) in zip(moge2_depth_list, moge1_depth_list, valid_mask_list)]

    # 计算全局平均缩放因子
    metric_scales_mean = torch.stack(metric_scales_list).mean().item()

    return moge1_depth_list, metric_scales_mean
    

if __name__ == "__main__":
    
    scene_metric_scales_mean = {}

    scene_ids = os.listdir(scene_path)

    for scene_id in tqdm(scene_ids):
        try:
            folder_name = os.path.join(depth_path, scene_id) + "/"
            exist = os.path.exists(folder_name)
            if not exist:
                os.makedirs(folder_name)
            else:
                continue
            image_names = [os.path.join(scene_path, scene_id, img_name) for img_name in os.listdir(os.path.join(scene_path, scene_id))]
            image_names.sort()
            images_read = []
            for image_name in image_names:
                images_read.append(Image.open(image_name))
            image_size = images_read[0].size
            resize = torchvision.transforms.Resize((image_size[1], image_size[0]))

            with torch.no_grad():
                images = load_and_preprocess_images(images_read).to(device) # 所有图像都会 resize 到 518
                with torch.cuda.amp.autocast(dtype=dtype):
                    # Predict attributes including cameras, depth maps, and point maps.
                    predictions = vggt_model(images)
                    depths_vggt = predictions["depth"][0, ...]
                    depth_vggt_conf = predictions["depth_conf"][0, ...]
            
            depths_vggt = resize(depths_vggt.permute(0, 3, 1, 2).contiguous())
            depth_vggt_conf = resize(depth_vggt_conf.unsqueeze(-1).permute(0, 3, 1, 2).contiguous())

            depths_moge1, depths_moge2 = [], []
            for input_image in images_read:
                # Read the input image and convert to tensor (3, H, W) with RGB values normalized to [0, 1]
                input_image = torch.tensor(np.array(input_image) / 255., dtype=torch.float32, device=device).permute(2, 0, 1)
                with torch.no_grad():
                    # Infer 
                    output = moge1_model.infer(input_image)
                    depth1 = output["depth"]
                    output = moge2_model.infer(input_image)
                    depth2 = output["depth"]
                    depths_moge1.append(depth1)
                    depths_moge2.append(depth2)
            depths_moge1 = torch.stack(depths_moge1)
            # depths_moge2 = torch.stack(depths_moge2)

            ### align MoGe-v1 depth to VGGT
            depth_list, valid_mask_list, all_valid_max_list = get_valid_depth(depths_vggt, depth_vggt_conf, depths_moge1, images_read)

            # 计算所有帧的有效最大值的中位数
            valid_max_array = np.array(all_valid_max_list)
            q50 = np.quantile(valid_max_array, 0.50)  # 计算50%分位点
            filtered_max = valid_max_array[valid_max_array <= q50]  # 过滤超过分位点的异常值
            
            # 取过滤后数据的最大值（正常范围内的最大值）
            global_avg_max = np.max(filtered_max)
            max_sky_value = global_avg_max * 5
            max_sky_value = np.minimum(max_sky_value, 1000)    # 相对深度最远不能超过 1000

            # 统一设置所有帧的无效区域值
            for i, (depth, valid_mask) in enumerate(zip(depth_list, valid_mask_list)):
                depth[~valid_mask] = max_sky_value
                
                # # 统计超限点占比（在clamp之前）
                # over_count = torch.sum(depth > max_sky_value).item()
                # total_pixels = depth.numel()
                # over_ratio = over_count / total_pixels * 100
                
                depth = torch.clamp(depth, max=max_sky_value)
                depth_list[i] = depth  # 更新处理后的深度图
            ###

            ### align moge depth and camera pose to metric depth (using MoGe-v2)
            align_metric_depths, metric_scales_mean = align_metric_depth(depth_list, depths_moge2, valid_mask_list)
            scene_metric_scales_mean[scene_id] = metric_scales_mean
            ###

            folder_name = os.path.join(depth_path, scene_id) + "/"
            os.makedirs(os.path.join(depth_path, scene_id), exist_ok=True)
            
            for image_name, depth_target in zip(image_names, align_metric_depths):
                depth_target = depth_target.cpu().numpy()
                depth_target = (depth_target - depth_target.min()) / (depth_target.max() - depth_target.min())
                target_image_path = os.path.join(folder_name, image_name.split("/")[-1])
                cv2.imwrite(target_image_path, (depth_target * 65535).astype(np.uint16))
            
            torch.cuda.empty_cache()
        except:
            print(f"scene_{scene_id} failed.")

    path = args.list.split("/")[-1].replace(".pkl", "")
    with open(f"scene_metric_scales_mean_{path}.json", "w") as f:
        json.dump(scene_metric_scales_mean, f)
