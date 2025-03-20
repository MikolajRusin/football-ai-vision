'''
*******************************************************************
*******************************************************************
*******************************************************************
*******************************************************************
'''

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import importlib
import utils.load_dataset
importlib.reload(utils.load_dataset)
import models.DETA_swin_model
importlib.reload(models.DETA_swin_model)

from models.DETA_swin_model import DETASwinConfig, DETASwinModel
from utils.load_dataset import LoadDataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from typing import Tuple, List, Dict
import albumentations as A
import numpy as np
import argparse

# Paths
PROJECT_PATH = Path(__file__).resolve().parent.parent   # Main project path
DATA_PATH = PROJECT_PATH / 'data'                       # Main data dir
IMAGES_PATH = DATA_PATH / 'images'                      # Images dir
TRAIN_SET_PATH = IMAGES_PATH / 'train'                  # Train set path
VALID_SET_PATH = IMAGES_PATH / 'valid'                  # Valid set path
COCO_ANNOTATIONS_PATH = DATA_PATH / 'coco_annotations'  # Path in which the COCO Annotations we saved
TRAIN_COCO_PATH = COCO_ANNOTATIONS_PATH / 'train.json'  # Train COCO JSON file path
VALID_COCO_PATH = COCO_ANNOTATIONS_PATH / 'valid.json'  # Valid COCO JSON file path

# ------------------------------------------------CUSTOM COLLATE FUNCTION----------------------------------------------------
def collate_fn(batch: List[Tuple[np.ndarray, Dict[np.ndarray, np.ndarray]]]) -> Tuple[List[np.ndarray], List[Dict[np.ndarray, np.ndarray]]]:
    '''
    Custom collate function to process batches of images and their correspondinng targets for a DataLoader.

    Args:
        batch (List[Tuple[np.ndarray, Dict[np.ndarray, np.ndarray]]]):
            A list where each element is a tuple containing:
            - Image (np.ndarray)
            - Dictionary containing information about bounding boxes and class labels.

    Returns:
        Tuple[List[np.ndarray], List[Dict[np.ndarray, np.ndarray]]]: 
            - A list of images converted to float32.
            - A list of corresponding target dictionaries.
    '''

    images, targets = zip(*batch)  # Extract the batch to the list of images and target annotations
    images = [img.astype(np.float32) for img in images]  # Convert images to type np.float32
    
    return images, targets
# ------------------------------------------------CUSTOM COLLATE FUNCTION----------------------------------------------------

# --------------------------------------LOAD DATASET LOADERS FOR TRAIN AND VALID IMAGES--------------------------------------
def load_dataloader(set_ratio: float, batch_size: int, transform_func: A.Compose=None) -> Tuple[DataLoader, DataLoader]:
    '''
    Loads the training and valid dataset with specified set ratio, batch size and transform function.

    Args:
        set_ratio (float): Ratio for selecting a subset of the dataset (0.0 - 1.0).
        batch_size (int): The number of iamges in one batch.
        transform_func (A.Compose): Function for data augmentation. Defaults to None.

    Returns:
        Tuple[DataLoader, DataLoader]: DataLoader objects for train and valid dataset.
    '''

    train_dataset = LoadDataset(         # Loading train dataset
        root_dir=TRAIN_SET_PATH,            # train direcotry path
        coco_json_file=TRAIN_COCO_PATH,     # train COCO JSON file path 
        transform=transform_func,           # function for data augmentation
        set_ratio=set_ratio,                # set ratio for data loading
        return_box_format='yolo',           # desire bounding box format
        normalize_bboxes=True               # normalize bounding boxes
    )
    train_dataloader = DataLoader(       # DataLoader object
        train_dataset,                      # train dataset
        batch_size=batch_size,              # number of images in one batch
        shuffle=False,                      # shuffle the images
        collate_fn=collate_fn               # custom collate function
    )

    valid_dataset = LoadDataset(         # Loading valid dataset
        root_dir=VALID_SET_PATH,            # valid direcotry path
        coco_json_file=VALID_COCO_PATH,     # valid COCO JSON file path 
        transform=None,                     # without data augmentation for valid dataset
        set_ratio=set_ratio,                # set ratio for data loading
        return_box_format='yolo',           # desire bounding box format
        normalize_bboxes=True               # normalize bounding boxes
    )
    valid_dataloader = DataLoader(       # DataLoader object
        valid_dataset,                      # valid dataset
        batch_size=batch_size,              # number of images in one batch
        shuffle=False,                      # shuffle the images
        collate_fn=collate_fn               # custom collate function
    )

    return train_dataloader, valid_dataloader
# --------------------------------------LOAD DATASET LOADERS FOR TRAIN AND VALID IMAGES--------------------------------------


# -----------------------------------------------------------MAIN------------------------------------------------------------
if __name__ == '__main__':
    # Load arguments from terminal
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs. Defaults to 10.')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size. Defaults to 2.')
    parser.add_argument('--set_ratio', type=float, default=None, help='Ratio of the datasets (0.0 - 1.0] or an integer for number of images. Defaults to None.')
    parser.add_argument('--backbone_lr', type=float, default=None, help="Learning rate for the backbone of the model. If None, the model will use the default learning rate ('lr') for all parameters. Defaults to None.")
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate for all model parameters, or for the non-backbone layers if 'backbone_lr' is set. Defaults to 0.0001.")
    # parser.add_argument('--val_frequency', type=float, default=50, help='Frequency of model validation.')
    # parser.add_argument('--checkpoints', type=bool, default=False, help='If True then the model saves checkpoints.')
    # parser.add_argument('--max_checkpoints', type=int, default=10, help='Maximum number of checkpoints contained in a folder.')
    # parser.add_argument('--use_scheduler', type=bool, default=False, help='If True the use learning rate scheduler (CosineAnnealingLR).')
    # parser.add_argument('--T_max', type=int, default=5, help='Number of epochs to reach minimum learning rate -> if use_scheduler=True.')
    # parser.add_argument('--eta_min', type=float, default=1e-6, help='Minimum learning rate after T_max epochs -> if use_scheduler=True.')
    args = parser.parse_args()

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2)
    ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))

    train_dataloader, valid_dataloader = load_dataloader(set_ratio=args.set_ratio, batch_size=args.batch_size, transform_func=transform)

    ID2LABEL = {
        0: 'N/A',
        1: 'Ball', 
        2: 'Goalkeeper', 
        3: 'Player', 
        4: 'Referee'
    }

    model_config = DETASwinConfig(
        model_id='jozhang97/deta-swin-large',
        id2label=ID2LABEL,
        optimizer=AdamW,
        backbone_lr=0.00001,
        lr=0.0001
    )

    model = DETASwinModel(config=model_config)
    print('The model has been loaded')
    print('****Start Training****')
    model.train_model(train_dataloader, valid_dataloader, num_epochs=args.num_epochs)


# -----------------------------------------------------------MAIN------------------------------------------------------------
