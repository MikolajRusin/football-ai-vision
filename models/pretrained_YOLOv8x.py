import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import supervision as sv
import cv2
from utils.box_utils import xcycwh_to_xyxy, xyxy_to_xywh, apply_nms
from dataclasses import dataclass
from typing import Union, Optional, List, Dict
from inference import get_model
from tqdm import tqdm

@dataclass
class YOLOModel:
    api_key: str
    model_id: str

    def __post_init__(self):
        self.model = get_model(api_key=self.api_key, model_id=self.model_id)

    def infer(self, data: Union[List[np.ndarray], np.ndarray], conf_threshold: float, iou_threshold: Optional[float] = None) -> List[sv.Detections]:
        outputs = self.model.infer(data, confidence=conf_threshold)
        outputs = self._process_outputs(outputs, iou_threshold=iou_threshold)

        return outputs
    
    def _process_outputs(self, outputs: List, iou_threshold: Optional[float] = None) -> List[sv.Detections]:
        all_boxes = [np.array([[pred.x, pred.y, pred.width, pred.height] for pred in output.predictions]) for output in outputs]
        all_scores = [np.array([pred.confidence for pred in output.predictions]) for output in outputs]
        all_class_ids = [np.array([pred.class_id for pred in output.predictions]) for output in outputs]

        processed_outputs = []
        for boxes_per_image, scores_per_image, class_ids_per_image in zip(all_boxes, all_scores, all_class_ids):
            boxes_per_image = xcycwh_to_xyxy(boxes_per_image)

            if iou_threshold is not None:
                boxes_per_image, scores_per_image, class_ids_per_image = apply_nms(boxes_per_image, scores_per_image, class_ids_per_image, 
                                                                                   iou_threshold=iou_threshold)
            detections = sv.Detections(
                xyxy=boxes_per_image,
                confidence=scores_per_image,
                class_id=class_ids_per_image
            ) 
            processed_outputs.append(detections)

        return processed_outputs
    
    def yolo_to_coco(self, yolo_preds: List[sv.Detections], img_paths: List[Path], categories: List[Dict]) -> Dict:
        coco_data = {
            'info': {'year': 2025, 'description': 'Football Player Detection'},
            'images': [],
            'annotations': [],
            'categories': categories
        }
        
        num_imgs = len(yolo_preds)
        zipped_data = zip(yolo_preds, img_paths)
        annotation_id = 1

        for img_id, (detections, img_path) in enumerate(tqdm(zipped_data, total=num_imgs, desc='Converting YOLO format to COCO format')):
            img = cv2.imread(img_path)
            height, width, _ = img.shape

            coco_data['images'].append({
                'id': img_id + 1,
                'file_name': img_path.name,
                'width': width,
                'height': height
            })

            for box, score, class_id in zip(detections.xyxy, detections.confidence, detections.class_id):
                x_min, y_min, width, height = xyxy_to_xywh(box.reshape(-1, 4))[0]
                bbox_xywh = [float(x_min), float(y_min), float(width), float(height)]

                coco_data['annotations'].append({
                    'id': annotation_id,
                    'image_id': img_id + 1,
                    'category_id': int(class_id + 1),
                    'bbox': bbox_xywh,
                    'area': float(width * height),
                    'iscrowd': 0,
                    'score': float(score)
                })
                annotation_id += 1

        return coco_data