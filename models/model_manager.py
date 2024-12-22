# models/model_manager.py
import torch
import torch.distributed as dist
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Any, Dict, Optional, Tuple
from ..config import TrainingConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)

class ModelManager:
    """Manages model loading, saving, and initialization."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self._setup_environment()
        
    def _setup_environment(self) -> None:
        """Setup distributed environment if needed."""
        if self.config.distributed:
            dist.init_process_group(backend="nccl")
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0
            
    def load_model(self, model_name: str, **kwargs) -> PreTrainedModel:
        """Load and prepare model for training."""
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.config.fp16 else torch.float32,
            **kwargs
        )
        
        if self.config.distributed:
            model = DDP(
                model,
                device_ids=[self.rank],
                output_device=self.rank,
                find_unused_parameters=True
            )
        
        return model.to(self.config.device)
    
    def load_tokenizer(self, model_name: str) -> PreTrainedTokenizer:
        """Load tokenizer for the model."""
        return AutoTokenizer.from_pretrained(model_name)
    
    def save_checkpoint(
        self,
        model: PreTrainedModel,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        step: int,
        metrics: Dict[str, float],
        path: str
    ) -> None:
        """Save training checkpoint."""
        if self.rank == 0:  # Only save on main process
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "step": step,
                "metrics": metrics,
                "config": self.config.__dict__
            }
            torch.save(checkpoint, path)
            logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(
        self,
        path: str,
        model: Optional[PreTrainedModel] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None
    ) -> Tuple[PreTrainedModel, Dict[str, Any]]:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device)
        
        if model is not None:
            model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
        return model, checkpoint

