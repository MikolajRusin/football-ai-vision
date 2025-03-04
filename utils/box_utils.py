import torch
import numpy as np
from typing import Tuple, Union
import torch
from supervision.detection.overlap_filter import box_non_max_suppression

def apply_nms(
        boxes: Union[np.ndarray, torch.Tensor], 
        scores: Union[np.ndarray, torch.Tensor], 
        class_ids: Union[np.ndarray, torch.Tensor],
        iou_threshold: float=0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    if len(boxes) == 0:
        return (
            np.array([]),
            np.array([]),
            np.array([])
        )
    
    # Convert to tensor
    if isinstance(boxes, np.ndarray):
        boxes = torch.Tensor(boxes)
    if isinstance(scores, np.ndarray):
        scores = torch.Tensor(scores)
    if isinstance(class_ids, np.ndarray):
        class_ids = torch.Tensor(class_ids)

    predictions = np.hstack((boxes, scores.reshape(-1, 1)))
    nms_indices = box_non_max_suppression(predictions=predictions, iou_threshold=iou_threshold)

    return (
        boxes[nms_indices].numpy(),
        scores[nms_indices].numpy(),
        class_ids[nms_indices].numpy().astype(int)
    )

def xywh_to_xyxy(boxes: np.ndarray) -> Tuple[np.ndarray]:
    x_min = boxes[:, 0] - (boxes[:, 2] / 2)
    y_min = boxes[:, 1] - (boxes[:, 3] / 2)
    x_max = boxes[:, 0] + (boxes[:, 2] / 2)
    y_max = boxes[:, 1] + (boxes[:, 3] / 2)

    return np.stack(
        [x_min, y_min, x_max, y_max], 
        axis=1
    )
