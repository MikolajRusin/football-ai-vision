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
    '''
    A class for loading model and performing inference with a YOLO detection model.

    Attributes:
        api_key (str): ROBOFLOW API Key for accessing the YOLO model.
        model_id (str): ID for the YOLO model.
    '''

    api_key: str
    model_id: str

    def __post_init__(self):
        '''
        Initializes the YOLO model using the provided API Key and model ID
        '''
        self.model = get_model(api_key=self.api_key, model_id=self.model_id)

    def infer(self, data: Union[List[np.ndarray], np.ndarray], conf_threshold: float, iou_threshold: Optional[float] = None) -> List[sv.Detections]:
        '''
        Runs inference on the input data.

        Args:
            data (Union[List[np.ndarray], np.ndarray]): Input images as a numPy array.
            conf_threshold (float): Confidence threshold for model detection.
            iou_threshold (Optional[float]): IoU threshold for Non-Maximum Suppression. Defaults to None.

            Returns:
                List[sv.Detections]: Processed detection results.
        '''

        outputs = self.model.infer(data, confidence=conf_threshold)
        outputs = self._process_outputs(outputs, iou_threshold=iou_threshold)

        return outputs
    
    def _process_outputs(self, outputs: List, iou_threshold: Optional[float] = None) -> List[sv.Detections]:
        '''
        Processes raw model outputs by converting coordinates and applying NMS if iou_threshold provided.

        Args:
            outputs (List): Raw model outputs containing detection predictions.
            iou_threshold (Optional[float]): IoU threshold for NMS. If None, NMS is not applied. Defaults to None.

        Returns:
            List[sv.Detections]: Processed detection results with bounding boxes, categories and scores.
        '''

        # Get all boxes, scores and class ids from the output
        all_boxes = [np.array([[pred.x, pred.y, pred.width, pred.height] for pred in output.predictions]) for output in outputs]
        all_scores = [np.array([pred.confidence for pred in output.predictions]) for output in outputs]
        all_class_ids = [np.array([pred.class_id for pred in output.predictions]) for output in outputs]

        processed_outputs = []  # List of containing processeed outputs

        # Process the outputs
        for boxes_per_image, scores_per_image, class_ids_per_image in zip(all_boxes, all_scores, all_class_ids):
            # Convert xcycwh (x_center, y_center, width, height) to xyxy (x_min, y_min, x_max, y_max) format
            boxes_per_image = xcycwh_to_xyxy(boxes_per_image)

            # Apply Non-Maximum Suppression
            if iou_threshold is not None:
                boxes_per_image, scores_per_image, class_ids_per_image = apply_nms(boxes_per_image, scores_per_image, class_ids_per_image, 
                                                                                   iou_threshold=iou_threshold)
            # Define the Detections object from boxes, scores and class ids
            detections = sv.Detections(
                xyxy=boxes_per_image,         # boxes in xyxy format
                confidence=scores_per_image,  # scores
                class_id=class_ids_per_image  # class_ids
            )

            processed_outputs.append(detections)

        return processed_outputs
    
    def yolo_to_coco(self, yolo_preds: List[sv.Detections], img_paths: List[Path], categories: List[Dict]) -> Dict:
        '''
        Converts YOLO detection results to COCO format.

        Args:
            yolo_preds (List[sv.Detections]): List of detections from YOLO model.
            img_paths (List[Path]): List of image paths corresponding to the detections.
            categories (List[Dict]): List of category mapping in COCO format.

        Returns:
            Dict: COCO-formatted dataset containing images metadata and annotations.
        '''

        # Define dictionary of image data info
        coco_data = {
            'info': {'year': 2025, 'description': 'Football Player Detection'},
            'images': [],
            'annotations': [],
            'categories': categories
        }
        
        num_imgs = len(yolo_preds)
        zipped_data = zip(yolo_preds, img_paths)
        annotation_id = 1

        # Process every image from yolo_preds
        for img_id, (detections, img_path) in enumerate(tqdm(zipped_data, total=num_imgs, desc='Converting YOLO format to COCO format')):
            img = cv2.imread(img_path)  # Read the image
            height, width, _ = img.shape  # Height and width of the image

            # Add the image info to coco images
            coco_data['images'].append({
                'id': img_id + 1,
                'file_name': img_path.name,
                'width': width,
                'height': height
            })

            # Process every box, score and class id for the image
            for box, score, class_id in zip(detections.xyxy, detections.confidence, detections.class_id):
                # Convert  xyxy (x_min, y_min, x_max, y_max) to xywh (x_min, y_min, width, height) (Appropriate format for COCO)
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