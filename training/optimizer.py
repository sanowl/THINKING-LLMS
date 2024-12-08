rom torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from typing import List, Tuple

def create_optimizer(model, config):
    """Create optimizer with weight decay."""
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params