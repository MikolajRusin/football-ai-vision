'''
Annotate Train, Valid and Test sets

As mentioned earlier, most of the transformers used in this project require xyxy format stored in the COCO JSON file, 
In this section we will convert the predictions from the YOLO model to the appropriate format and then save them to JSON FILE in COCO format.
'''
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.pretrained_YOLOv8x import YOLOModel
from typing import Generator
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from math import ceil
import argparse
import os
import json
import cv2

# ----------------------------------------------------CREATE BATCH----------------------------------------------------
def create_batch(data: list, batch_size: int) -> Generator[list, None, None]:
    '''
    Generate batches from an input data sequence with a specified batch size.

    Args:
        data (list): The list of input data sequence to be batched.
        batch_size (int): The number of elements in each batch.

    Yields:
        Generator[list, None, None]: A generator yielding batches of the input sequence.
    '''

    # List of the current batch
    current_batch = []

    # Iterate through the input data
    for element in data:
        # If length of the current batch reaches the specified batch size then yield it
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = [] # Reset batch

        current_batch.append(element)

    # Yield the last batch if it contains any elements
    if current_batch:
        yield current_batch
# ----------------------------------------------------CREATE BATCH----------------------------------------------------


# ---------------------------------------------------------MAIN-------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_threshold', type=float, default=0.3, help='Confidence threshold for model predictions')
    parser.add_argument('--iou_threshold', type=float, default=None, help='IoU threshold for Non-Maximum Supression')
    parser.add_argument('--batch_size', type=int, default=4, help='Number of elements in each batch')
    args = parser.parse_args()

    # Paths
    PROJECT_PATH = Path(__file__).resolve().parent.parent # Path of the project
    DATA_PATH = PROJECT_PATH / 'data'  # Main data dir
    IMAGES_PATH = DATA_PATH / 'images'  # Images dir
    TRAIN_SET_PATH = IMAGES_PATH / 'train'  # Train set path
    VALID_SET_PATH = IMAGES_PATH / 'valid'  # Valid set path
    TEST_SET_PATH = IMAGES_PATH / 'test'  # Test set path
    COCO_ANNOTATIONS_PATH = DATA_PATH / 'coco_annotations'
    os.makedirs(COCO_ANNOTATIONS_PATH, exist_ok=True)

    load_dotenv()
    ROBOLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')
    ROBOFLOW_MODEL_ID = 'football-players-detection-3zvbc/11'
    CONF_THRESHOLD = args.conf_threshold
    IOU_THRESHOLD = args.iou_threshold
    BATCH_SIZE = args.batch_size
    DATA_ANNOTATION_MODEL = YOLOModel(
        api_key=ROBOLOW_API_KEY, model_id=ROBOFLOW_MODEL_ID,
        conf_threshold=CONF_THRESHOLD, iou_threshold=IOU_THRESHOLD
    )
    
    CATEGORIES = [
        {'id': 1, 'name': 'ball'},
        {'id': 2, 'name': 'goalkeeper'},
        {'id': 3, 'name': 'player'},
        {'id': 4, 'name': 'referee'}
    ]

    dataset_paths = [
        TRAIN_SET_PATH,
        VALID_SET_PATH,
        TEST_SET_PATH
    ]

    for dataset_path in dataset_paths:
        img_file_names = os.listdir(dataset_path)

        annotated_img_paths = []
        annotated_imgs = []
        for img_file_names_batch in tqdm(create_batch(img_file_names, batch_size=BATCH_SIZE), 
                                         total=ceil(len(img_file_names) / BATCH_SIZE), 
                                         desc=f'Annotating images in {dataset_path} | Total: {len(img_file_names)} | Batch size: {BATCH_SIZE}'):
            
            img_paths = [dataset_path / img_file_name for img_file_name in img_file_names_batch]
            img_batch = [cv2.imread(dataset_path / img_path) for img_path in img_paths]
            img_batch = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in img_batch]

            output = DATA_ANNOTATION_MODEL.infer(img_batch)
            annotated_img_paths.extend(img_paths)
            annotated_imgs.extend(output)

        coco_data = DATA_ANNOTATION_MODEL.yolo_to_coco(annotated_imgs, annotated_img_paths, categories=CATEGORIES)

        with open(COCO_ANNOTATIONS_PATH / f'{dataset_path.name}.json', 'w') as json_file:
            json.dump(coco_data, json_file, indent=4)
# ---------------------------------------------------------MAIN-------------------------------------------------------