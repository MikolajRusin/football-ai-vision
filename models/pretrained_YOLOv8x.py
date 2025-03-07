import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils import box_utils
from dataclasses import dataclass
from inference import get_model



@dataclass
class YOLOModel:
    api_key: str
    model_id: str
    conf_threshold: float
    iou_threshold: float
    print('dasdasfasfsaf')

    def __post_init__(self):
        self.model = get_model(api_key=self.api_key, model_id=self.model_id)

