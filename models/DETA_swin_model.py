import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from transformers.models.deprecated.deta.modeling_deta import DetaObjectDetectionOutput
from transformers import DetaForObjectDetection, DetaImageProcessor
from torch.optim import AdamW, Optimizer
from dataclasses import dataclass
from typing import Optional, List, Dict
from torch import nn
import torch
import numpy as np

# ---------------------------------------------------------DETA Swin Model Config-------------------------------------------------------
@dataclass
class DETASwinConfig:
    '''
    Configuration class for DETA Swin Large model
    
    Attributes:
        model_id (str): ID of the pretrained model to use. Defaults to 'jozhang97/deta-swin-large'.
        optimizer (torch.optim.Optimizer): The optimizerused for training model. It updates the model's weights based on the computed gradient. Based to AdamW.
        backbone_lr (float):
        fc_layer_lr (float):
        id2label (Optional[Dict[int, str]]): Mapping of class IDs to labels. If None, the default COCO labels will be used. Deafults to None.
    '''

    model_id: str = 'jozhang97/deta-swin-large'
    optimizer: Optimizer = AdamW
    backbone_lr: float = 0.00001
    fc_layer_lr: float = 0.0001
    id2label: Optional[Dict[int, str]] = None
# ---------------------------------------------------------DETA Swin Model Config-------------------------------------------------------


# -------------------------------------------------------------DETA Swin Model----------------------------------------------------------
@dataclass
class DETASwinModel(nn.Module):
    '''
    Class for the DETA Swin Large object detection model.

    This class load a pretrained DETA Swin Large model and prepares it for inference.
    Class also allows to fine tune the model.
    Attributes:

    '''
    config: DETASwinConfig

    def __post_init__(self):
        # Define the device to which the model and data will be assigned
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Define model and processor object
        self.processor = DetaImageProcessor.from_pretrained(self.config.model_id, use_fast=True)
        self.model = DetaForObjectDetection.from_pretrained(self.config.model_id)
        
        if self.config.id2label is not None:
            label2id = {v: k for k, v in self.config.id2label.items()}
            self.model.config.id2label = self.config.id2label
            self.model.config.label2id = label2id
            self.model.config.num_labels = len(label2id)

        self.model.to(self.device)


    def forward(self, imgs: List[np.ndarray], targets: Optional[List[Dict[np.ndarray, np.ndarray]]]=None):
        '''
        Forward pass of the model.
        
        Args:
            imgs (List[np.ndarray]): List of images, each image with shape (height, width, channels). A list should be a batch of images (in np.ndarray format) with length batch_size.
            targets: (Optional[List[Dict[np.ndarray, np.ndarray]]]): List of dictionaries containing 'bboxes' (np.ndarray, shape (N, 4)) and 'class_labels' (np.ndarray, shape (N,)). 
                                                                     The length of the list should correspond to the length of imgs. Defaults to None

        Returns:
            DetaObjectDetectionOutput: The model's prediction output containing logits, predicted boxes, etc.
        '''

        inputs = self.processor(images=imgs, return_tensors='pt').to(self.device)

        if targets is not None:
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            outputs = self.model(**inputs, labels=targets)
        else:
            outputs = self.model(**inputs)

        return outputs
# -------------------------------------------------------------DETA Swin Model----------------------------------------------------------

