'''
*******************************************************************
*******************************************************************
*******************************************************************
*******************************************************************
'''

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from transformers.models.deprecated.deta.modeling_deta import DetaObjectDetectionOutput
from transformers import DetaForObjectDetection, DetaImageProcessor
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from torch import nn
from typing import Optional, List, Dict
from dataclasses import dataclass
from tqdm import tqdm
import torch
import numpy as np
import wandb

# ---------------------------------------------------------DETA Swin Model Config-------------------------------------------------------
@dataclass
class DETASwinConfig:
    '''
    Configuration class for DETA Swin Large model
    
    Attributes:
        model_id (str): ID of the pretrained model to use. Defaults to 'jozhang97/deta-swin-large'.
        id2label (Optional[Dict[int, str]]): Mapping of class IDs to labels. If None, the default COCO labels will be used. Deafults to None.
        optimizer (torch.optim.Optimizer): The optimizerused for training model. It updates the model's weights based on the computed gradient. Based to AdamW.
        backbone_lr (float): Learning rate for the backbone of the model. If None, the model will use the default learning rate ('lr') for all parameters. Defaults to None.
        lr (float): Learning rate for all model parameters, or for the non-backbone layers if 'backbone_lr' is set. Defaults to 0.0001.
    '''

    model_id: str = 'jozhang97/deta-swin-large'
    id2label: Optional[Dict[int, str]] = None
    optimizer: Optimizer = AdamW
    backbone_lr: float = None
    lr: float = 0.0001
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
        super().__init__()
        # Define the device to which the model and data will be assigned
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Define model and processor object
        self.processor = DetaImageProcessor.from_pretrained(self.config.model_id, use_fast=True)
        self.model = DetaForObjectDetection.from_pretrained(self.config.model_id)
        
        # Adjust class names and class ids to suit our needs
        if self.config.id2label is not None:
            self._configure_model_class_ids()

        # Create and configure optimzier for training the model
        self.optimizer = self._configure_optimizer()

        # Move the model to the defined device
        self.model.to(self.device)

    def forward(self, imgs: List[np.ndarray], targets: Optional[List[Dict[np.ndarray, np.ndarray]]]=None) -> DetaObjectDetectionOutput:
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
    
    def train_model(self, train_dataloader: DataLoader, valid_dataloader: DataLoader, num_epochs: 10) -> None:
        # Initialize wandb to create a workspace for track metrics
        wandb.init(
            project='DETA_swin_player_detection',
            name='test_dasad'
        )

        # Training loop for epoch
        for epoch in range(num_epochs):
            self._train_one_epoch(train_dataloader, valid_dataloader, epoch, num_epochs)

        wandb.finish()

    def _train_one_epoch(self, train_dataloader: DataLoader, valid_dataloader: DataLoader, epoch: int, num_epochs: int) -> any:
        # Training loop for iteration in epoch
        for iteration, (img_batch, target_batch) in enumerate(tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}')):
            self.model.train()


    def _configure_model_class_ids(self) -> None:
        '''
        Configured model class IDs.

        This function sets up the custom class IDs and their corresponding labels in the model's config id2labels and labels2id.
        The function also changes the number of labels in the model.
        '''

        label2id = {v: k for k, v in self.config.id2label.items()}  # Converts labels to their corresponding IDs
        self.model.config.id2label = self.config.id2label           # Set the custom id2label
        self.model.config.label2id = label2id                       # Set the custom label2id
        self.model.config.num_labels = len(label2id)                # Change the number of labels

    def _configure_optimizer(self) -> Optimizer:
        '''
        Configures the optimizer for the model.

        This function sets up the optimizer with different learning rates for the backbone and thre rest of the model's parameters. 
        If 'backbone_lr' from model's config is specified, the function assigns the different learning rate to the backbone and a different to the rest of the model's parameters.
        Otherwise, a single learning rate ('lr) is applied to all parameters.

        Returns:
            Optimizer: The configured optimizer with specified parameters for training the model.
        '''

        if self.config.backbone_lr is not None:
            # Extract the backbone parameters required for gradients
            backbone_params = [
                param
                for name, param in self.model.named_parameters() if 'backbone' in name and param.requires_grad
            ]
            # Extract the rest of model's parameters required for gradients
            non_backbone_params = [
                param
                for name, param in self.model.named_parameters() if 'backbone' not in name and param.requires_grad
            ]

            # Create the list of optimizer parameters
            optimizer_params = [
                {'params': backbone_params, 'lr': self.config.backbone_lr},
                {'params': non_backbone_params, 'lr': self.config.lr}
            ]
        else:
            # Extract all model's parameters required for gradients
            model_params = [
                param
                for param in self.model.named_parameters() if param.requires_grad
            ]

            # Create the list of optimizer parameters
            optimizer_params = [
                {'params': model_params, 'lr': self.config.lr}
            ]

        # Define the specified optimizer with early extracted parameters 
        optimizer = self.config.optimizer(optimizer_params)

        return optimizer
# -------------------------------------------------------------DETA Swin Model----------------------------------------------------------

