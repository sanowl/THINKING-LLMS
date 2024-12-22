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

