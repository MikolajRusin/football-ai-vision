'''
This Python file contains utility functions for working with bounding boxes in object detection tasks.
It includes implementations for:

1. **Non-Maximum Supression (NMS)** A Method to remove redundant averlapping bounding boxes based on their scores.
2. **Normalize bboxes** A method to normalize bounding boxes.
3. **Denormalizes bboxes** A method to denormalize bounding boxes.
4. **Bounding Box Format Conversions**:
    - xcycwh_to_xyxy
    - xywh_to_xcycwh
    - xywh_to_xyxy
    - xyxy_to_xywh

These functiosns are useful for processing object detection outputs from deepl learning models.
'''

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

# -------------------------------------------------------NORMALIZE BBOXES-------------------------------------------------------
def normalize_bboxes(boxes: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
    '''
    Normalizes bounding boxes to range (0.0 - 1.0). Normalization is carried out on the basis of the given width and height 

    Args:
        boxes (np.ndarray): Bounding boxes to normalize, shape (N, 4).
        image_shape (Tuple[int, int]): Original image width and height.

    Returns:
        np.ndarray: Normalized bounding boxes, shape (N, 4).
    '''

    # Normalize bounding boxes
    x_normalized = boxes[:, [0, 2]] / image_shape[1]  # X coordinates / width
    y_normalized = boxes[:, [1, 3]] / image_shape[0]  # Y coordinates / width

    return np.stack(
        [x_normalized[:, 0], y_normalized[:, 0], x_normalized[:, 1], y_normalized[:, 1]], 
        axis=1
    )
# -------------------------------------------------------NORMALIZE BBOXES-------------------------------------------------------

# ------------------------------------------------------DENORMALIZE BBOXES------------------------------------------------------
def denormalize_bboxes(boxes: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
    '''
    Denormalizes bounding boxes from range (0.0 - 1.0) back to original values. 

    Args:
        boxes (np.ndarray): Bounding boxes to normalize, shape (N, 4).
        image_shape (Tuple[int, int]): Original image width and height.

    Returns:
        np.ndarray: Denormalized bounding boxes, shape (N, 4).
    '''

    # Denormalize bounding boxes
    x_denormalized = boxes[:, [0, 2]] * image_shape[1]  # X coordinates * width
    y_denormalized = boxes[:, [1, 3]] * image_shape[0]  # Y coordinates * width

    # Ensure that the bounding box coordinates are withing image boundaries
    x_denormalized = np.clip(x_denormalized, 0, image_shape[1])  # Clip y values to [0, width]
    y_denormalized = np.clip(y_denormalized, 0, image_shape[0])  # Clip x values to [0, height]

    # Round the nearest integer
    x_denormalized = np.round(x_denormalized).astype(float)
    y_denormalized = np.round(y_denormalized).astype(float)

    return np.stack(
        [x_denormalized[:, 0], y_denormalized[:, 0], x_denormalized[:, 1], y_denormalized[:, 1]], 
        axis=1
    )
# ------------------------------------------------------DENORMALIZE BBOXES------------------------------------------------------

# -------------------------------------------------CONVERT XCYCWH TO XYXY FORMAT------------------------------------------------
def xcycwh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    '''
    Converts bounding boxes from xcycwh (x_center, y_center, width, height) to xyxy (x_min, y_min, x_max, y_max) format.

    Args:
        boxes (np.ndarray): Bounding boxes in xywh format, shape (N, 4).

    Returns:
        np.ndarray: Bounding boxes in xyxy format, shape (N, 4).
    '''

    x_c, y_c, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    x_min = x_c - (width / 2)   # x_center - width / 2
    y_min = y_c - (height / 2)  # y_center - height / 2
    x_max = x_c + (width / 2)   # x_center + width / 2
    y_max = y_c + (height / 2)  # y_center + height / 2

    return np.stack(
        [x_min, y_min, x_max, y_max], 
        axis=1
    )
# --------------------------------------------------CONVERT XCYCWH TO XYXY FORMAT------------------------------------------------

# --------------------------------------------------CONVERT XYWH TO XCYCWH FORMAT------------------------------------------------
def xywh_to_xcycwh(boxes: np.ndarray) -> np.ndarray:
    '''
    Converts bounding boxes from xywh (x_min, y_min, width, height) to xcycwh (x_center, y_center, width, height) format.

    Args:
        boxes (np.ndarray): Bounding boxes in xywh format, shape (N, 4).

    Returns:
        np.ndarray: Bounding boxes in xcycwh format, shape (N, 4).
    '''

    x_min, y_min, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    x_c = x_min + (width / 2)   # x_min + width / 2  (x_c -> X coordiantes of the center object)
    y_c = y_min + (height / 2)  # y_min + height / 2  (y_c -> Y coordiantes of the center object)

    return np.stack(
        [x_c, y_c, width, height], 
        axis=1
    )
# --------------------------------------------------CONVERT XYWH TO XCYCWH FORMAT------------------------------------------------

# ---------------------------------------------------CONVERT XYWH TO XYXY FORMAT-------------------------------------------------
def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    '''
    Converts bounding boxes from xywh (x_min, y_min, width, height) to xyxy (x_min, y_min, x_max, y_max) format.

    Args:
        boxes (np.ndarray): Bounding boxes in xywh format, shape (N, 4).

    Returns:
        np.ndarray: Bounding boxes in xyxy format, shape (N, 4).
    '''

    x_min, y_min, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    x_max = x_min + width   # x_min + width
    y_max = y_min + height  # y_min + height

    return np.stack(
        [x_min, y_min, x_max, y_max], 
        axis=1
    )
# ---------------------------------------------------CONVERT XYWH TO XYXY FORMAT-------------------------------------------------

# ---------------------------------------------------CONVERT XYXY TO XYWH FORMAT-------------------------------------------------
def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    '''
    Converts bounding boxes from xyxy (x_min, y_min, x_max, y_max) to xywh (x_min, y_min, width, height) format.

    Args:
        boxes (np.ndarray): Bounding boxes in xyxy format, shape (N, 4).

    Returns:
        np.ndarray: Bounding boxes in xywh format, shape (N, 4).
    '''
    
    x_min, y_min, x_max, y_max = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    width = x_max - x_min   # Width of the boxes
    height = y_max - y_min  # Height of the boxes

    return np.stack(
        [x_min, y_min, width, height],
        axis=1
    )
# ---------------------------------------------------CONVERT XYXY TO XYWH FORMAT-------------------------------------------------