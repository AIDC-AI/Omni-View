import re
import json
from collections import defaultdict
import numpy as np
import argparse
import string
import torch
from loguru import logger as eval_logger

from tqdm import tqdm
from typing import Union
from scipy.spatial.transform import Rotation as R
from pytorch3d.ops import box3d_overlap
from pytorch3d.transforms import euler_angles_to_matrix


def rotation_3d_in_euler(points, angles, return_mat=False, clockwise=False):
    """Rotate points by angles according to axis.

    Args:
        points (np.ndarray | torch.Tensor | list | tuple ):
            Points of shape (N, M, 3).
        angles (np.ndarray | torch.Tensor | list | tuple):
            Vector of angles in shape (N, 3)
        return_mat: Whether or not return the rotation matrix (transposed).
            Defaults to False.
        clockwise: Whether the rotation is clockwise. Defaults to False.

    Raises:
        ValueError: when the axis is not in range [0, 1, 2], it will
            raise value error.

    Returns:
        (torch.Tensor | np.ndarray): Rotated points in shape (N, M, 3).
    """
    batch_free = len(points.shape) == 2
    if batch_free:
        points = points[None]

    if len(angles.shape) == 1:
        angles = angles.expand(points.shape[:1] + (3, ))
        # angles = torch.full(points.shape[:1], angles)

    assert len(points.shape) == 3 and len(angles.shape) == 2 \
        and points.shape[0] == angles.shape[0], f'Incorrect shape of points ' \
        f'angles: {points.shape}, {angles.shape}'

    assert points.shape[-1] in [2, 3], \
        f'Points size should be 2 or 3 instead of {points.shape[-1]}'

    rot_mat_T = euler_angles_to_matrix(angles, 'ZXY')  # N, 3,3
    rot_mat_T = rot_mat_T.transpose(-2, -1)

    if clockwise:
        raise NotImplementedError('clockwise')

    if points.shape[0] == 0:
        points_new = points
    else:
        points_new = torch.bmm(points, rot_mat_T)

    if batch_free:
        points_new = points_new.squeeze(0)

    if return_mat:
        if batch_free:
            rot_mat_T = rot_mat_T.squeeze(0)
        return points_new, rot_mat_T
    else:
        return points_new


class EulerDepthInstance3DBoxes:
    """3D boxes of instances in Depth coordinates.

    We keep the "Depth" coordinate system definition in MMDet3D just for
    clarification of the points coordinates and the flipping augmentation.

    Coordinates in Depth:

    .. code-block:: none

                    up z    y front (alpha=0.5*pi)
                       ^   ^
                       |  /
                       | /
                       0 ------> x right (alpha=0)

    The relative coordinate of bottom center in a Depth box is (0.5, 0.5, 0),
    and the yaw is around the z axis, thus the rotation axis=2.
    The yaw is 0 at the positive direction of x axis, and decreases from
    the positive direction of x to the positive direction of y.
    Also note that rotation of DepthInstance3DBoxes is counterclockwise,
    which is reverse to the definition of the yaw angle (clockwise).

    Attributes:
        tensor (torch.Tensor): Float matrix of N x box_dim.
        box_dim (int): Integer indicates the dimension of a box
            Each row is (x, y, z, x_size, y_size, z_size, alpha, beta, gamma).
        with_yaw (bool): If True, the value of yaw will be set to 0 as minmax
            boxes.
    """

    def __init__(self,
                 tensor,
                 box_dim=9,
                 with_yaw=True,
                 origin=(0.5, 0.5, 0.5)):

        if isinstance(tensor, torch.Tensor):
            device = tensor.device
        else:
            device = torch.device('cpu')
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that
            # does not depend on the inputs (and consequently confuses jit)
            tensor = tensor.reshape((0, box_dim)).to(dtype=torch.float32,
                                                     device=device)
        assert tensor.dim() == 2 and tensor.size(-1) == box_dim, tensor.size()

        if tensor.shape[-1] == 6:
            # If the dimension of boxes is 6, we expand box_dim by padding
            # (0, 0, 0) as a fake euler angle.
            assert box_dim == 6
            fake_rot = tensor.new_zeros(tensor.shape[0], 3)
            tensor = torch.cat((tensor, fake_rot), dim=-1)
            self.box_dim = box_dim + 3
        elif tensor.shape[-1] == 7:
            assert box_dim == 7
            fake_euler = tensor.new_zeros(tensor.shape[0], 2)
            tensor = torch.cat((tensor, fake_euler), dim=-1)
            self.box_dim = box_dim + 2
        else:
            assert tensor.shape[-1] == 9
            self.box_dim = box_dim
        self.tensor = tensor.clone()

        self.origin = origin
        if origin != (0.5, 0.5, 0.5):
            dst = self.tensor.new_tensor((0.5, 0.5, 0.5))
            src = self.tensor.new_tensor(origin)
            self.tensor[:, :3] += self.tensor[:, 3:6] * (dst - src)
        self.with_yaw = with_yaw

    def __len__(self) -> int:
        """int: Number of boxes in the current object."""
        return self.tensor.shape[0]

    def __getitem__(self, item: Union[int, slice, np.ndarray, torch.Tensor]):
        """
        Args:
            item (int or slice or np.ndarray or Tensor): Index of boxes.

        Note:
            The following usage are allowed:

            1. `new_boxes = boxes[3]`: Return a `Boxes` that contains only one
               box.
            2. `new_boxes = boxes[2:10]`: Return a slice of boxes.
            3. `new_boxes = boxes[vector]`: Where vector is a
               torch.BoolTensor with `length = len(boxes)`. Nonzero elements in
               the vector will be selected.

            Note that the returned Boxes might share storage with this Boxes,
            subject to PyTorch's indexing semantics.

        Returns:
            :obj:`BaseInstance3DBoxes`: A new object of
            :class:`BaseInstance3DBoxes` after indexing.
        """
        original_type = type(self)
        if isinstance(item, int):
            return original_type(self.tensor[item].view(1, -1),
                                 box_dim=self.box_dim,
                                 with_yaw=self.with_yaw)
        b = self.tensor[item]
        assert b.dim() == 2, \
            f'Indexing on Boxes with {item} failed to return a matrix!'
        return original_type(b, box_dim=self.box_dim, with_yaw=self.with_yaw)

    @property
    def dims(self) -> torch.Tensor:
        """Tensor: Size dimensions of each box in shape (N, 3)."""
        return self.tensor[:, 3:6]

    @classmethod
    def overlaps(cls, boxes1, boxes2, mode='iou', eps=1e-4):
        """Calculate 3D overlaps of two boxes.

        Note:
            This function calculates the overlaps between ``boxes1`` and
            ``boxes2``, ``boxes1`` and ``boxes2`` should be in the same type.

        Args:
            boxes1 (:obj:`EulerInstance3DBoxes`): Boxes 1 contain N boxes.
            boxes2 (:obj:`EulerInstance3DBoxes`): Boxes 2 contain M boxes.
            mode (str): Mode of iou calculation. Defaults to 'iou'.
            eps (bool): Epsilon. Defaults to 1e-4.

        Returns:
            torch.Tensor: Calculated 3D overlaps of the boxes.
        """
        assert isinstance(boxes1, EulerDepthInstance3DBoxes)
        assert isinstance(boxes2, EulerDepthInstance3DBoxes)
        assert type(boxes1) == type(boxes2), '"boxes1" and "boxes2" should' \
            f'be in the same type, got {type(boxes1)} and {type(boxes2)}.'

        assert mode in ['iou']

        rows = len(boxes1)
        cols = len(boxes2)
        if rows * cols == 0:
            return boxes1.tensor.new(rows, cols)

        corners1 = boxes1.corners
        corners2 = boxes2.corners
        _, iou3d = box3d_overlap(corners1, corners2, eps=eps)
        return iou3d

    @property
    def corners(self):
        """torch.Tensor: Coordinates of corners of all the boxes
        in shape (N, 8, 3).

        Convert the boxes to corners in clockwise order, in form of
        ``(x0y0z0, x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0)``

        .. code-block:: none

                                           up z
                            front y           ^
                                 /            |
                                /             |
                  (x0, y1, z1) + -----------  + (x1, y1, z1)
                              /|            / |
                             / |           /  |
               (x0, y0, z1) + ----------- +   + (x1, y1, z0)
                            |  /      .   |  /
                            | / origin    | /
               (x0, y0, z0) + ----------- + --------> right x
                                          (x1, y0, z0)
        """
        if self.tensor.numel() == 0:
            return torch.empty([0, 8, 3], device=self.tensor.device)

        dims = self.dims
        corners_norm = torch.from_numpy(
            np.stack(np.unravel_index(np.arange(8), [2] * 3),
                     axis=1)).to(device=dims.device, dtype=dims.dtype)

        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
        # use relative origin
        assert self.origin == (0.5, 0.5, 0.5), \
            'self.origin != (0.5, 0.5, 0.5) needs to be checked!'
        corners_norm = corners_norm - dims.new_tensor(self.origin)
        corners = dims.view([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])

        # rotate
        corners = rotation_3d_in_euler(corners, self.tensor[:, 6:])

        corners += self.tensor[:, :3].view(-1, 1, 3)
        return corners


def transform_scanrefer_bbox(bbox, extrinsic=None):
    center = bbox[0: 3]
    sizes = bbox[3: 6]
    rot = R.from_euler("zxy", np.array(bbox[6:9]))
    if extrinsic is not None:
        center = (extrinsic @ np.array([*center, 1]).reshape(4, 1)).reshape(4)[:3].tolist()
        mat = extrinsic[:3, :3] @ rot.as_matrix()
        rot = R.from_matrix(mat)
    zxy = list(rot.as_euler("zxy"))

    return center + sizes + zxy


def proposal_matching(proposals, pred_bbox):
    ret = pred_bbox
    max_iou = 0
    for proposal in proposals:
        cur_bbox = proposal + [0, 0, 0]
        try:
            iou = EulerDepthInstance3DBoxes.overlaps(
                EulerDepthInstance3DBoxes(torch.tensor([pred_bbox])),
                EulerDepthInstance3DBoxes(torch.tensor([cur_bbox]))
            ).item()
        except Exception as e:
            print(f"Error in calculating IoU: {e}")
            continue
        if iou > max_iou:
            max_iou = iou
            ret = cur_bbox
    return ret


def scanrefer_process_results(doc, results, proposals):
    gt_bbox = doc["gt_bbox"]
    lines = results.strip('\n').strip("```").strip("json").strip("\n").split("\n")
    pred_dict = None
    for line in lines:
        if "bbox_3d" in line:
            try:
                pred_dict = eval(line.strip())
            except Exception as e:
                eval_logger.error(f"Error parsing bbox_3d: {line.strip()}")
            break
    
    iou = 0
    pred_bbox = None
    if pred_dict is not None:
        # if len(doc["cam2global"]) != 32:
        #     return 1
        assert "frame" in pred_dict and isinstance(pred_dict["frame"], int) and pred_dict["frame"] >= 0 and pred_dict["frame"] < len(doc["cam2global"]), \
            "Invalid frame index"
        assert "bbox_3d" in pred_dict and isinstance(pred_dict["bbox_3d"], list) and len(pred_dict["bbox_3d"]) == 9, \
            "Invalid bbox_3d format"
        
        frame_idx = pred_dict["frame"]
        extrinsic = np.array(doc["axis_align_matrix"]) @ np.array(doc["cam2global"][frame_idx])
        # pred_bbox = transform_scanrefer_bbox((np.array(pred_dict["bbox_3d"]) / 100.).tolist(), extrinsic)
        pred_bbox = transform_scanrefer_bbox(pred_dict["bbox_3d"], extrinsic)
        refined_bbox = proposal_matching(proposals, pred_bbox)
        iou = EulerDepthInstance3DBoxes.overlaps(
            EulerDepthInstance3DBoxes(torch.tensor([refined_bbox])),
            EulerDepthInstance3DBoxes(torch.tensor([gt_bbox]))
        ).item()

    return iou


def main(args):
    with open(args.input_file) as f:
        data = json.load(f)
    
    with open("/mnt/workspace/cv_multimodal/aigc/dataset/video_3d_llm_data/data/metadata/scannet_val_pred_box.json") as f:
        scan2obj = json.load(f)
    
    iou25_acc_per_type = defaultdict(list)
    iou50_acc_per_type = defaultdict(list)

    for item in tqdm(data):
        proposals = scan2obj[item["scene"]]
        iou = scanrefer_process_results(item['gt_response'], item['pred_response'], proposals)

        iou25_acc_per_type["all"].append(iou >= 0.25)
        iou50_acc_per_type["all"].append(iou >= 0.5)
        iou25_acc_per_type[item["question_type"]].append(iou >= 0.25)
        iou50_acc_per_type[item["question_type"]].append(iou >= 0.5)
    
    for k in iou25_acc_per_type:
        print(f"{k} iou@0.25: {np.mean(iou25_acc_per_type[k]) * 100}")
        print(f"{k} iou@0.5: {np.mean(iou50_acc_per_type[k]) * 100 }")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, default='results/scanrefer/val.jsonl')
    parser.add_argument("-n", type=int, default=-1)
    args = parser.parse_args()

    main(args)