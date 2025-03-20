'''
Python file responsible for managing
'''


from pathlib import Path
from dataclasses import dataclass
from collections import deque
import torch
import os

@dataclass
class CheckpointManager:
    '''
    A class to manage model checkpoints with a maximum limit.

    This class handles saving, deleting and managing model checkpoints in a FIFO (First In First Out) queue.
    When the maximum number of checkpoints is reached, the oldest checkpoint is automatically deleted
    before saving a new one.

    Attributes:
        checkpoints_dir_path (Path): Path to the directory where the checkpoints will be stored.
        max_checkpoints (int): Maximum number of checkpoint files in the directory. Defaults to 10.
    '''

    checkpoint_dir_path: Path
    max_checkpoints: int = 10

    def __post_init__(self):
        '''
        Function to initialize checkpoints FIFO (First In First Out) queue.
        '''

        self.checkpoint_dir_path.mkdir(parents=True, exist_ok=True)           # Make direcotry if does not exist
        old_checkpoints = os.listdir(self.checkpoint_dir_path)                # Load old checkpoints if their exist
        self.checkpoint_queue = deque(old_checkpoints, self.max_checkpoints)  # Define FIFO queue with old checkpoints if their exists, 
                                                                              # otherwise the queue will be empty

    def save_checkpoint(self, checkpoint, checkpoint_name: str) -> None:
        '''
        Saves a new checkpoint and manages the total number of checkpoints.

        Args:
            checkpoint: The model checkpoint to save.
            checkpoint_name (str): Name of the checkpoint to save.
        '''

        # If the queue length is equal to number of max checkpoints, 
        # delete the last checkpoint before writing the next one
        if len(self.checkpoint_queue) == self.max_checkpoints:
            self._delete_last_checkpoint()
        
        # Save the new checkpoint
        new_checkpoint_path = self.checkpoint_dir_path / checkpoint_name
        torch.save(checkpoint, new_checkpoint_path)
        print(f'Checkpoint {checkpoint_name} saved as the path: {new_checkpoint_path}')
        
        # Add the new checkpoint to the queue
        self.checkpoint_queue.append(checkpoint_name)

    def _delete_last_checkpoint(self) -> None:
        '''
        Deletes the oldest checkpoint from the directory and removes it from tracking.
        '''

        last_checkpoint_name = self.checkpoint_queue.popleft()
        last_checkpoint_path = self.checkpoint_dir_path / last_checkpoint_name
        os.remove(last_checkpoint_path)
