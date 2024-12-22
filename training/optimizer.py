# training/optimizer.py (continued)
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from typing import List, Tuple, Any
import torch

def create_optimizer(model: torch.nn.Module, config: Any) -> torch.optim.Optimizer:
    """Create optimizer with weight decay."""
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                      if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                      if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return AdamW(optimizer_grouped_parameters, lr=config.learning_rate)

def create_scheduler(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    config: Any
) -> Any:
    """Create learning rate scheduler."""
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=num_training_steps
    )

