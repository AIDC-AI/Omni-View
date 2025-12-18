# Copyright 2025 Bytedance Ltd. and/or its affiliates.

# Modified by AIDC-AI, 2025
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, 
# software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, 
# either express or implied. See the License for the specific language governing permissions and limitations under the License.

import os
import json
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision
import cv2
cv2.setNumThreads(8)

from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

from data.transforms import ImageTransform
from data.data_utils import pil_img2rgb, add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.bagel.qwen2_navit import NaiveCache
from data.pose_utils import compute_plucker_embedding_batch

from modeling.autoencoder import load_ae


# from VGGT: https://github.com/facebookresearch/vggt/blob/main/vggt/utils/geometry.py
def depth_to_cam_coords_points(depth_map: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a depth map to camera coordinates.

    Args:
        depth_map (np.ndarray): Depth map of shape (H, W).
        intrinsic (np.ndarray): Camera intrinsic matrix of shape (3, 3).

    Returns:
        tuple[np.ndarray, np.ndarray]: Camera coordinates (H, W, 3)
    """
    H, W = depth_map.shape
    assert intrinsic.shape == (3, 3), "Intrinsic matrix must be 3x3"
    assert intrinsic[0, 1] == 0 and intrinsic[1, 0] == 0, "Intrinsic matrix must have zero skew"

    # Intrinsic parameters
    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]

    # Generate grid of pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Unproject to camera coordinates
    x_cam = (u - cu) * depth_map / fu
    y_cam = (v - cv) * depth_map / fv
    z_cam = depth_map

    # Stack to form camera coordinates
    cam_coords = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    return cam_coords


def closed_form_inverse_se3(se3, R=None, T=None):
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

    If `R` and `T` are provided, they must correspond to the rotation and translation
    components of `se3`. Otherwise, they will be extracted from `se3`.

    Args:
        se3: Nx4x4 or Nx3x4 array or tensor of SE3 matrices.
        R (optional): Nx3x3 array or tensor of rotation matrices.
        T (optional): Nx3x1 array or tensor of translation vectors.

    Returns:
        Inverted SE3 matrices with the same type and device as `se3`.

    Shapes:
        se3: (N, 4, 4)
        R: (N, 3, 3)
        T: (N, 3, 1)
    """
    # Check if se3 is a numpy array or a torch tensor
    is_numpy = isinstance(se3, np.ndarray)

    # Validate shapes
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    # Extract R and T if not provided
    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        T = se3[:, :3, 3:]  # (N,3,1)

    # Transpose R
    if is_numpy:
        # Compute the transpose of the rotation for NumPy
        R_transposed = np.transpose(R, (0, 2, 1))
        # -R^T t for NumPy
        top_right = -np.matmul(R_transposed, T)
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_transposed = R.transpose(1, 2)  # (N,3,3)
        top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
        inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
        inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix


def depth_to_world_coords_points(
    depth_map: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    eps=1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a depth map to world coordinates.

    Args:
        depth_map (np.ndarray): Depth map of shape (H, W).
        intrinsic (np.ndarray): Camera intrinsic matrix of shape (3, 3).
        extrinsic (np.ndarray): Camera extrinsic matrix of shape (3, 4).

    Returns:
        tuple[np.ndarray, np.ndarray]: World coordinates (H, W, 3) and valid depth mask (H, W).
    """
    if depth_map is None:
        return None, None, None

    # Valid depth mask
    point_mask = depth_map > eps

    # Convert depth map to camera coordinates
    cam_coords_points = depth_to_cam_coords_points(depth_map, intrinsic)

    # Multiply with the inverse of extrinsic matrix to transform to world coordinates
    # extrinsic_inv is 4x4 (note closed_form_inverse_OpenCV is batched, the output is (N, 4, 4))
    cam_to_world_extrinsic = closed_form_inverse_se3(extrinsic[None])[0]

    R_cam_to_world = cam_to_world_extrinsic[:3, :3]
    t_cam_to_world = cam_to_world_extrinsic[:3, 3]

    # Apply the rotation and translation to the camera coordinates
    world_coords_points = np.dot(cam_coords_points, R_cam_to_world.T) + t_cam_to_world  # HxWx3, 3x3 -> HxWx3
    # world_coords_points = np.einsum("ij,hwj->hwi", R_cam_to_world, cam_coords_points) + t_cam_to_world

    return world_coords_points


def render_from_cameras_videos(points, colors, extrinsics, intrinsics, height, width):
    
    homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
    
    render_list = []
    mask_list = []
    depth_list = []
    # Render from each camera
    for frame_idx in range(len(extrinsics)):
        # Get corresponding camera parameters
        extrinsic = extrinsics[frame_idx]
        intrinsic = intrinsics[frame_idx]
        
        camera_coords = (extrinsic @ homogeneous_points.T).T[:, :3]
        projected = (intrinsic @ camera_coords.T).T
        uv = projected[:, :2] / projected[:, 2].reshape(-1, 1)
        depths = projected[:, 2]    
        
        pixel_coords = np.round(uv).astype(int)  # pixel_coords (h*w, 2)      
        valid_pixels = (  # valid_pixels (h*w, )      valid_pixels is the valid pixels in width and height
            (pixel_coords[:, 0] >= 0) & 
            (pixel_coords[:, 0] < width) & 
            (pixel_coords[:, 1] >= 0) & 
            (pixel_coords[:, 1] < height)
        )
        
        pixel_coords_valid = pixel_coords[valid_pixels]  # (h*w, 2) to (valid_count, 2)
        colors_valid = colors[valid_pixels]
        depths_valid = depths[valid_pixels]
        uv_valid = uv[valid_pixels]
        
        valid_mask = (depths_valid > 0) & (depths_valid < 60000) # & normal_angle_mask
        colors_valid = colors_valid[valid_mask]
        depths_valid = depths_valid[valid_mask]
        pixel_coords_valid = pixel_coords_valid[valid_mask]

        # Initialize depth buffer
        depth_buffer = np.full((height, width), np.inf)
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # Vectorized depth buffer update
        if len(pixel_coords_valid) > 0:
            rows = pixel_coords_valid[:, 1]
            cols = pixel_coords_valid[:, 0]
                            
            # Sort by depth (near to far)
            sorted_idx = np.argsort(depths_valid)
            rows = rows[sorted_idx]
            cols = cols[sorted_idx]
            depths_sorted = depths_valid[sorted_idx]
            colors_sorted = colors_valid[sorted_idx]

            # Vectorized depth buffer update
            depth_buffer[rows, cols] = np.minimum(
                depth_buffer[rows, cols], 
                depths_sorted
            )
            
            # Get the minimum depth index for each pixel
            flat_indices = rows * width + cols  # Flatten 2D coordinates to 1D index
            unique_indices, idx = np.unique(flat_indices, return_index=True)
            
            # Recover 2D coordinates from flattened indices
            final_rows = unique_indices // width
            final_cols = unique_indices % width
            
            image[final_rows, final_cols] = colors_sorted[idx, :3].astype(np.uint8)

        mask = np.zeros_like(depth_buffer, dtype=np.uint8)
        mask[depth_buffer != np.inf] = 255
        
        render_list.append(image)
        mask_list.append(mask[..., None])
        depth_list.append(depth_buffer[..., None])
    
    return render_list, mask_list, depth_list


def get_mean_and_std(img):
    mean, std = cv2.meanStdDev(img)
    mean = np.hstack(np.around(mean, 2))
    std = np.hstack(np.around(std, 2))
    return mean, std


def color_transfer(source, target):
    source = cv2.cvtColor(source, cv2.COLOR_RGB2LAB)
    target = cv2.cvtColor(target, cv2.COLOR_RGB2LAB)
    s_mean, s_std = get_mean_and_std(source)
    t_mean, t_std = get_mean_and_std(target)
    image_new = (source - s_mean) * (t_std / s_std) + t_mean
    image_new = np.clip(image_new, 0, 255)
    dst = cv2.cvtColor(cv2.convertScaleAbs(image_new), cv2.COLOR_LAB2RGB)
    return dst


print('setup config')
model_path = "./pretrained_model/BAGEL-7B-MoT"  # Download from https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT

# LLM config preparing
llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
llm_config.qk_norm = True
llm_config.tie_word_embeddings = False
llm_config.layer_module = "Qwen2MoTDecoderLayer"

# ViT config preparing
vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
vit_config.rope = False
vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

# VAE loading
vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

# Bagel config preparing
config = BagelConfig(
    visual_gen=True,
    visual_und=True,
    llm_config=llm_config, 
    vit_config=vit_config,
    vae_config=vae_config,
    vit_max_num_patch_per_side=70,
    connector_act='gelu_pytorch_tanh',
    latent_patch_size=2,
    max_latent_size=64,
)

torch.set_default_device("cpu")
config.visual_rendered_nvs = True
language_model = Qwen2ForCausalLM(llm_config)
vit_model      = SiglipVisionModel(vit_config)
model          = Bagel(language_model, vit_model, config)
model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=False)

# Tokenizer Preparing
tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

# Image Transform Preparing
vae_transform = ImageTransform(640 * 2, 352 * 2, 16) # generation in grid, so it should be 2*2
vit_transform = ImageTransform(640, 384, 14)

max_mem_per_gpu = "80GiB"  # Modify it according to your GPU setting. On an A100, 80 GiB is sufficient to load on a single GPU.

device_map = infer_auto_device_map(
    model,
    max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
    no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
)

same_device_modules = [
    'language_model.model.embed_tokens',
    'time_embedder',
    'latent_pos_embed',
    'vae2llm',
    'llm2vae',
    'connector',
    'vit_pos_embed'
]

if torch.cuda.device_count() == 1:
    first_device = device_map.get(same_device_modules[0], "cuda:0")
    for k in same_device_modules:
        if k in device_map:
            device_map[k] = first_device
        else:
            device_map[k] = "cuda:0"
else:
    first_device = device_map.get(same_device_modules[0])
    for k in same_device_modules:
        if k in device_map:
            device_map[k] = first_device

print('start load model')
# Thanks @onion-liu: https://github.com/ByteDance-Seed/Bagel/pull/8
model = load_checkpoint_and_dispatch(
    model,
    checkpoint="model.safetensors",
    device_map=device_map,
    offload_buffers=True,
    dtype=torch.float32,
    force_hooks=True,
    offload_folder="/tmp/offload"
)
model = model.to(torch.bfloat16)
model = model.eval()


# Prepare depth estimation
import sys;sys.path.append("depth_pipe")
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from moge.model.v1 import MoGeModel as MoGeV1Model

device = "cuda"
dtype = torch.bfloat16
# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

# Load the model from huggingface hub (or load from local). 
moge1_model = MoGeV1Model.from_pretrained("Ruicheng/moge-vitl").to(device)

from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
skyseg_processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")
skyseg_model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large").to(device)
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


@torch.no_grad()
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


from inferencer import InterleaveInferencer

inferencer = InterleaveInferencer(
    model=model, 
    vae_model=vae_model, 
    tokenizer=tokenizer, 
    vae_transform=vae_transform, 
    vit_transform=vit_transform, 
    new_token_ids=new_token_ids
)

import random
import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

inference_hyper=dict(
    cfg_text_scale=1.0,
    cfg_img_scale=1.0,
    cfg_interval=[0.4, 1.0],
    timestep_shift=1.5,
    num_timesteps=50,
    cfg_renorm_min=0.0,
    cfg_renorm_type="global",
    image_shapes=(352, 640),
)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--scene-id", required=True, help="The scene id in RE10k.")
parser.add_argument("--pose-id", default=None, help="The camera trajectory in RE10k.")
parser.add_argument("--image-path", default=None, help="Provided image.")
args = parser.parse_args()

scene_id = args.scene_id
pose_id = args.pose_id or scene_id

if scene_id is not None:
    image_path = f"/data/cpfs_0/jkhu/test/images/{scene_id}/00000.png"
else:
    image_path = args.image_path
    scene_id = image_path.split("/")[-1].split(".")[0]
    assert pose_id is not None, "Please give a pose id."

os.makedirs(f"results/{scene_id}", exist_ok=True)

camera_pose_id = f"/data/cpfs_0/jkhu/test/metadata/{pose_id}.json"
with open(camera_pose_id) as camera_pose_file:
    camera_pose_data = json.load(camera_pose_file)["frames"]

for chunk_size in [1, 2, 3, 4]:
    image_indices = list(range((chunk_size - 1) * 12, chunk_size * 12 + 1))
    reference_idx = str(image_indices[0]).zfill(5)
    print(f"chunk {chunk_size}. image_indices: {image_indices}, reference_idx: {reference_idx}")

    if chunk_size == 1:
        target_image = np.array(Image.open(image_path).resize((640, 360)))
    else:
        target_image = np.array(Image.open(f"results/{scene_id}/{reference_idx}.png").resize((640, 360)))

    width, height = Image.open(image_path).size
    if chunk_size == 1:
        previous_images = [Image.open(image_path).convert("RGB").resize((640, 360))]
    else:
        assert output_image is not None
        # 取右下作为 reference , 但是在最后crop存图像的时候已经把右下提出来了，所以直接用output_image
        previous_images = [output_image]

    print("render")
    # get intrinsics and extrinsics
    intrinsics, extrinsics = [], []
    for index in image_indices:
        cur_frame = camera_pose_data[index]
        fx, fy, cx, cy = cur_frame["fxfycxcy"]

        # NOTE CHECK resize in generation, the intrinsics should be updated
        actual_resize_scale = height / 352
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        intrinsic[:2, :] = intrinsic[:2, :] * actual_resize_scale

        intrinsics.append(intrinsic)
        extrinsics.append(np.array(cur_frame["w2c"]))
    intrinsics = np.stack(intrinsics)
    extrinsics = np.stack(extrinsics)

    # estimate depth map via VGGT and MoGe
    # TODO(hujiakui): it can be estimated by the geometry module
    if chunk_size == 1:
        images_read = [Image.open(image_path).resize((640, 360))]
    else:
        images_read = [Image.open(f"results/{scene_id}/{reference_idx}.png").resize((640, 360))]

    with torch.inference_mode():
        images = load_and_preprocess_images(images_read).to(device) # resize 到 518
        with torch.cuda.amp.autocast(dtype=dtype):
            # Predict attributes including cameras, depth maps, and point maps.
            predictions = vggt_model(images)
            depths_vggt = predictions["depth"][0, ...]
            depth_vggt_conf = predictions["depth_conf"][0, ...]
        
        resize = torchvision.transforms.Resize((360, 640))
        depths_vggt = resize(depths_vggt.permute(0, 3, 1, 2).contiguous())
        depths_vggt_conf = resize(depth_vggt_conf.unsqueeze(-1).permute(0, 3, 1, 2).contiguous())
        
        input_image = torch.tensor(np.array(images_read[0]) / 255., dtype=torch.float32, device=device).permute(2, 0, 1)
        output = moge1_model.infer(input_image)
        depth1 = output["depth"].unsqueeze(0)

        depth_list, valid_mask_list, all_valid_max_list = get_valid_depth(depths_vggt, depths_vggt_conf, depth1, images_read)
        depth = depth_list[-1]
        valid_mask = valid_mask_list[-1]

        # 计算所有帧的有效最大值的中位数
        valid_max_array = np.array(all_valid_max_list)
        q50 = np.quantile(valid_max_array, 0.50)  # 计算50%分位点
        filtered_max = valid_max_array[valid_max_array <= q50]  # 过滤超过分位点的异常值
        
        # 取过滤后数据的最大值（正常范围内的最大值）
        global_avg_max = np.max(filtered_max)
        max_sky_value = global_avg_max * 5
        max_sky_value = np.minimum(max_sky_value, 1000)    # 相对深度最远不能超过 1000

        depth[~valid_mask] = max_sky_value
        depth = torch.clamp(depth, max=max_sky_value)

        depth = depth.cpu().numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        depth = (depth * 65535).astype(np.uint16)

    depth = np.array(Image.fromarray(depth).resize((640, 360)))

    # Backproject point cloud
    point_map = depth_to_world_coords_points(depth, extrinsics[0], intrinsics[0])
    points = point_map.reshape(-1, 3)
    colors = np.array(previous_images[0].resize((640, 360))).reshape(-1, 3)

    # render
    render_list, mask_list, _ = render_from_cameras_videos(
        points, colors, extrinsics[1:], intrinsics[1:], height=height, width=width
    )

    # get rays plucker
    intrinsics = torch.from_numpy(intrinsics)
    extrinsics = torch.from_numpy(extrinsics)
    rays_plucker = compute_plucker_embedding_batch(352, width, intrinsics, extrinsics, device=intrinsics.device).to(torch.bfloat16)
    rays_plucker_list = rays_plucker.chunk(len(image_indices), dim=0)

    # merge 4 images to 2x2 image
    # .resize((640, 352))
    width, height = 640, 352
    merged_previous_images, merged_rays_plucker, merged_masked_condition_images = [], [], [None]
    for idx in range(len(image_indices) // 4 + 1):
        if idx == 0:
            # merge image
            merged_image = Image.new("RGB", (width * 2, height * 2))
            merged_image.paste(previous_images[0].resize((640, 352)), (0, 0))
            merged_image.paste(previous_images[0].resize((640, 352)), (width, 0))
            merged_image.paste(previous_images[0].resize((640, 352)), (0, height))
            merged_image.paste(previous_images[0].resize((640, 352)), (width, height))
            merged_previous_images.append(merged_image)
            # merge plucker ray
            merged_ray_plucker = torch.cat([rays_plucker_list[0]] * 4)
            merged_ray_plucker = merged_ray_plucker.view(2, 2, height, width, -1).contiguous().permute(0, 2, 1, 3, 4).flatten(0, 1).flatten(1, 2)
            merged_rays_plucker.append(merged_ray_plucker.unsqueeze(0))
        else:
            # merge plucker ray
            merged_ray_plucker = torch.cat(rays_plucker_list[(idx - 1) * 4 + 1: idx * 4 + 1])
            merged_ray_plucker = merged_ray_plucker.view(2, 2, height, width, -1).contiguous().permute(0, 2, 1, 3, 4).flatten(0, 1).flatten(1, 2)
            merged_rays_plucker.append(merged_ray_plucker.unsqueeze(0))
            # merge condition_rendered_images
            merged_masked_condition_image = Image.new("RGB", (width * 2, height * 2))
            merged_masked_condition_image.paste(Image.fromarray(render_list[(idx - 1) * 4 + 0]).resize((640, 352)), (0, 0))
            merged_masked_condition_image.paste(Image.fromarray(render_list[(idx - 1) * 4 + 1]).resize((640, 352)), (width, 0))
            merged_masked_condition_image.paste(Image.fromarray(render_list[(idx - 1) * 4 + 2]).resize((640, 352)), (0, height))
            merged_masked_condition_image.paste(Image.fromarray(render_list[(idx - 1) * 4 + 3]).resize((640, 352)), (width, height))
            merged_masked_condition_images.append(merged_masked_condition_image)
    
    previous_images = [merged_previous_images[0]]

    print("setup generate")
    h, w, p = 22 * 2, 40 * 2, 2
    # 对第一帧做处理
    masked_condition_images = [None]
    for frame_idx, render_image in enumerate(merged_masked_condition_images):
        if frame_idx == 0:
            continue
        masked_condition_latent = vae_model.encode(vae_transform(render_image).unsqueeze(0))
        # image = vae_model.decode(masked_condition_latent)
        # image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
        # image = Image.fromarray((image).to(torch.uint8).cpu().numpy())
        # image.save(f"results/{scene_id}/render_{frame_idx}.png")
        # del image

        masked_condition_latent = masked_condition_latent[:, :h * p, :w * p].reshape(model.latent_channel, h, p, w, p)
        masked_condition_latent = torch.einsum("chpwq->hwpqc", masked_condition_latent).reshape(-1, p * p * model.latent_channel)
        # HardCODE (640, 352) -> vae.encode() -> 3520 length -> patch_size = 2 -> 880
        masked_condition_images.append(masked_condition_latent)

    print("generate")
    color_fix = True
    for frame_idx in range(len(image_indices) // 4):
        output_dict = inferencer(
            text=None, 
            image=previous_images, 
            rays_plucker=merged_rays_plucker, 
            masked_condition_images=masked_condition_images, 
            **inference_hyper
        )
        output_image = output_dict['image']
        previous_images.append(output_image)

        if color_fix:
            output_image = np.array(output_image)
            output_image = Image.fromarray(color_transfer(output_image, target_image))
        
        output_images = [
            output_image.crop((0, 0, 640, 352)),
            output_image.crop((640, 0, 640 * 2, 352)),
            output_image.crop((0, 352, 640, 352 * 2)),
            output_image.crop((640, 352, 640 * 2, 352 * 2)),
        ]
        for image_idx, output_image in zip(range(image_indices[frame_idx * 4 + 1], image_indices[frame_idx * 4 + 1] + 4), output_images):
            image_name = str(image_idx).zfill(5)
            output_image.save(f"results/{scene_id}/{image_name}.png")
