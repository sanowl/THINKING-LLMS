# data/dataset.py
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import List, Dict

class ThoughtDataset(Dataset):
    """Dataset class for thought-response pairs."""
    
    def __init__(
        self,
        prompts: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int
    ):
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.prompts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        prompt = self.prompts[idx]
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze()
        }

# data/data_utils.py
from typing import List, Tuple
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True
) -> DataLoader:
    """Create a DataLoader with proper settings."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

def process_batch(
    batch: Dict[str, torch.Tensor],
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """Process a batch of data by moving it to the target device."""
    return {k: v.to(device) for k, v in batch.items()}

# data/__init__.py
from .dataset import ThoughtDataset
from .data_utils import create_dataloader, process_batch

__all__ = ['ThoughtDataset', 'create_dataloader', 'process_batch']