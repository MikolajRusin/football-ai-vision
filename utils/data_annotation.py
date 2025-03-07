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
import os
from dotenv import load_dotenv

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
    # Paths
    PROJECT_PATH = Path(__file__).resolve().parent.parent # Path of the project
    DATA_PATH = PROJECT_PATH / 'data'  # Main data dir
    IMAGES_PATH = DATA_PATH / 'images'  # Images dir
    TRAIN_SET_PATH = IMAGES_PATH / 'train'  # Train set path
    VALID_SET_PATH = IMAGES_PATH / 'valid'  # Valid set path
    TEST_SET_PATH = IMAGES_PATH / 'test'  # Test set path
    print(PROJECT_PATH)

    load_dotenv()
    ROBOLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')
    ROBOFLOW_MODEL_ID = 'football-players-detection-3zvbc/11'
    CONF_THRESHOLD = 0.3
    IOU_THRESHOLD = 0.5
    DATA_ANNOTATE_MODEL = YOLOModel(
        api_key=ROBOLOW_API_KEY, model_id=ROBOFLOW_MODEL_ID,
        conf_threshold=CONF_THRESHOLD, iou_threshold=IOU_THRESHOLD
    )

    data_sets = [
        TRAIN_SET_PATH,
        VALID_SET_PATH,
        TEST_SET_PATH
    ]
# ---------------------------------------------------------MAIN-------------------------------------------------------