'''
This Python file defines a PyTorch Dataset class used for loading and processing datasets from a specified directory and COCO JSON annotation file.

The class handles the loading of images and corresponding annotations, applies transformations, 
and supports filtering the dataset based on a ratio (`set_ratio`).

The bounding boxes can be returned in various formats such as such as 'xcycwh' or 'yolo', 'xyxy' or 'pascal_voc', 'xywh' or 'coco'.
'''

from utils.box_utils import normalize_bboxes, xywh_to_xcycwh, xywh_to_xyxy
from torch.utils.data import Dataset
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Tuple, Dict
import albumentations
import numpy as np
import os
import json
import cv2


@dataclass
class LoadDataset(Dataset):
    '''
    PyTorch Dataset class for loading and processing dataset from specified direcotry and COCO JSON file.

    Args:
        root_dir (Path): Path to the directory containing images.
        coco_json_file (Path): Path to the COCO JSON annotation file.
        transform (Union[albumentations.compose]): Transformaction to be applied to image and bounding boxes. Defaults to None.
        set_ratio (Union[float]): Ratio of dataset to use (0.0 < set_ratio <= 1.0). Defaults to None.
        return_box_format (str): Desired bounding box format, such as 'xcycwh' or 'yolo', 'xyxy' or 'pascal_voc', 'xywh' or 'coco'. Defaults to 'xcycwh'.

    Raises:
        ValueError: if set_ratio is outside the valid range.
        ValueError: if length of images in direcotry does not match the length of images in COCO JSON file.
        TypeError: if set_ratio is not a float.
        TypeError: if return_box_format is not recognized.
    '''

    root_dir: Path
    coco_json_file: Path
    transform: Union[albumentations.Compose] = None
    set_ratio: Union[float] = None
    return_box_format: str = 'xcycwh'

    def __post_init__(self):
        '''
        Initializes the dataset by loading image file names and COCO metadata.

        Raises:
            ValueError: if length of images in direcotry does not match the length of images in COCO JSON file.
        '''

        # Set the random seed
        np.random.seed(42)

        # Load image file names from prvided directory
        self.image_file_names = np.array(os.listdir(self.root_dir))
        # Shuffle images
        np.random.shuffle(self.image_file_names)
        # If set_ratio is provided, apply filtering to reduce the dataset size
        if self.set_ratio is not None:
            self._filter_images_by_ratio()

        # Load coco_data from provided JSON file
        with open(self.coco_json_file, 'r') as f:
            self.coco_data = json.load(f)
        # If set_ratio is provided, filter COCO data to include only selected images
        if self.set_ratio is not None:
            self._filter_coco_data()

        # Check if the length of images in direcotry is the same as the length of images in COCO JSON file.
        if not (self.image_file_names.shape[0] == len(self.coco_data['images'])):
            raise ValueError(
                f'The size of image file names from directory and images from COCO JSON file are not the same.'
                f"\nImages in directory -> {self.image_file_names.shape[0]}"
                f"\nImages in COCO JSON file -> {len(self.coco_data['images'])}"
            )
        
        print(f'Loaded {self.image_file_names.shape[0]} images from {self.root_dir}')

    def _filter_images_by_ratio(self) -> None:
        '''
        Fitlers the dataset to retain only subset of images based on 'set_ratio'.

        Raises:
            ValueError: If `set_ratio` is outside the valid range (0.0 < set_ratio <= 1.0).
            TypeError: If `set_ratio` is not a float.
        '''

        if isinstance(self.set_ratio, float):
            if 0.0 < self.set_ratio <= 1.0:
                num_images = int(self.image_file_names.shape[0] * self.set_ratio)
            else:
                raise ValueError(f'Invalid value for set_ratio. Expected range for set_ratio is 0.0 < set_ratio <= 1.0 but got {self.set_ratio}')
        else:
            raise TypeError(f'Invalid type of set_ratio. Expected type of set_ratio is float but got {type(self.set_ratio)}')
            
        self.image_file_names = self.image_file_names[:num_images]

    def _filter_coco_data(self) -> None:
        '''
        Filters the COCO dataset to retain only the images and annotations that match 'self.image_file_names' (If the 'set_ratio' is specified).
        '''

        filtered_imgs = [img_data
                         for img_data in self.coco_data['images']
                         if img_data['file_name'] in self.image_file_names]
        
        filtered_img_ids = set(img_data['id']
                               for img_data in filtered_imgs)
        
        filtered_img_ann = [ann
                            for ann in self.coco_data['annotations']
                            if ann['image_id'] in filtered_img_ids]
        
        self.coco_data['images'] = filtered_imgs
        self.coco_data['annotations'] = filtered_img_ann

    def __len__(self) -> int:
        '''
        Returns the number of images in the dataset.

        Returns:
            int: The total number of load images.
        '''

        return self.image_file_names.shape[0]
    
    def __getitem__(self, index) -> Tuple[np.ndarray, Dict[np.ndarray, np.ndarray]]:
        '''
        Retrieves an image and its corresponding bounding boxes and labels.

        Args:
            index (int): Index of the image in the loaded dataset.

        Returns:
            tuple: (image, target)
                - image (np.ndarray): The image in RGB format.
                - target (dict): A dictionary containing 'bboxes' (bounding boxes) and 'class_labels' (labels for the bounding boxes).

        Raises:
            TypeError: if return_box_format is not recognized.
        '''

        # Load the image
        img_file_name = self.image_file_names[index]
        img_path = self.root_dir / img_file_name
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Retrieve the image ID from coco_data corresponding with the selected image file name
        img_id = next((img_data['id'] for img_data in self.coco_data['images'] if img_data['file_name'] == img_file_name))

        # Initialize bounding boxes and labels
        bboxes = []
        labels = []
        
        # Select annotations for the corresponding image id
        for ann in self.coco_data['annotations']:
            if ann['image_id'] == img_id:
                bboxes.append(ann['bbox'])
                labels.append(ann['category_id'])

        # Convert to numpy arrays
        labels = np.array(labels)
        bboxes = np.array(bboxes)

        # Apply transformation if the transform function is provided
        if self.transform is not None:
            transformed = self.transform(image=img, bboxes=bboxes, class_labels=labels)      
            img = transformed['image']      # Transformed image
            bboxes = transformed['bboxes']  # Transformerd bounding boxes
            labels = transformed['class_labels']

        # Convert bounding boxes to the required format
        if self.return_box_format == 'xcycwh' or self.return_box_format == 'yolo':
            bboxes = xywh_to_xcycwh(bboxes)
        elif self.return_box_format == 'xyxy' or self.return_box_format == 'pascal_voc':
            bboxes = xywh_to_xyxy(bboxes)
        elif self.return_box_format == 'xywh' or self.return_box_format == 'coco':
            bboxes = bboxes
        else:
            raise TypeError(
                'Not recognized type of bounding box format.'
                f'\nProvided: {self.return_box_format}'
                "\nExpected: 'xcycwh' or 'yolo', 'xyxy' or 'pascal_voc', 'xywh' or 'coco'"
            )
        
        # Normalize bboxes to range [0.0 - 1.0]
        bboxes = normalize_bboxes(bboxes, img.shape)

        # Prepare target dictionary
        target = {
            'bboxes': bboxes,
            'class_labels': labels
        }

        return img, target
            