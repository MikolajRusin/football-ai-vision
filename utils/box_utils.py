import torch
import numpy as np
from typing import Tuple, Union
import torch

# ----------------------------------------------------NON-MAXIMUM SUPRESSION----------------------------------------------------
def apply_nms(
        boxes: Union[np.ndarray, torch.Tensor], 
        scores: Union[np.ndarray, torch.Tensor], 
        class_ids: Union[np.ndarray, torch.Tensor],
        iou_threshold: float=0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Applies Non-Maximum Suppression (NMS) to remove redundant bounding boxes.

    Args:
        boxes (np.ndarray ot torch.Tensor): Bounding boxes in xyxy (x_min, y_in, x_max, y_max) format, shape (N, 4).
        scores (np.ndarray or torch.Tensor): Confidence scores for each bounding box, shape (N,).
        class_ids (np.ndarray or torch.Tensor): Class IDs associated with each bounding box, shape (N,).
        iou_threshold (float): Intersection-over-Union (IoU) threshold for NMS. Defaults to 0,5.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Filtered bounding boxesm confidence scores and class IDs.
    '''

    if len(boxes) == 0:
        return (
            np.array([]),
            np.array([]),
            np.array([])
        )
    
    # Convert to tensor if necessary
    if isinstance(boxes, np.ndarray):
        boxes = torch.Tensor(boxes)
    if isinstance(scores, np.ndarray):
        scores = torch.Tensor(scores)
    if isinstance(class_ids, np.ndarray):
        class_ids = torch.Tensor(class_ids)

    # Apply NSM function
    nms_indices = torch.ops.torchvision.nms(boxes, scores, iou_threshold)

    return (
        boxes[nms_indices].numpy(),
        scores[nms_indices].numpy(),
        class_ids[nms_indices].numpy().astype(int)
    )
# ----------------------------------------------------NON-MAXIMUM SUPRESSION----------------------------------------------------

# ----------------------------------------------------CONVERT XYXY TO XYWH FORMAT----------------------------------------------------
def xywh_to_xyxy(boxes: np.ndarray) -> Tuple[np.ndarray]:
    '''
    Converts bounding boxes from xywh (x_center, y_center, width, height) to xyxy (x_min, y_min, x_max, y_max) format.

    Args:
        boxes (np.ndarray): Bounding boxes in xywh format, shape (N, 4).

    Returns:
        np.ndarray: Bounding boxes in xyxy format, shape (N, 4).
    '''

    x_min = boxes[:, 0] - (boxes[:, 2] / 2)  # x_center - width / 2
    y_min = boxes[:, 1] - (boxes[:, 3] / 2)  # y_center - height / 2
    x_max = boxes[:, 0] + (boxes[:, 2] / 2)  # x_center + width / 2
    y_max = boxes[:, 1] + (boxes[:, 3] / 2)  # y_center + height / 2

    return np.stack(
        [x_min, y_min, x_max, y_max], 
        axis=1
    )
# ----------------------------------------------------CONVERT XYXY TO XYWH FORMAT----------------------------------------------------