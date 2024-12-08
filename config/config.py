# config/config.py
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any
from datetime import datetime
import torch
import yaml
import numpy as np

@dataclass
class TrainingConfig:
    # Model configurations
    base_model_name: str
    judge_model_name: str
    sentence_model_name: str = "all-MiniLM-L6-v2"
    
    # Training parameters
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    warmup_steps: int = 1000
    max_steps: int = 100000
    eval_steps: int = 500
    save_steps: int = 1000
    
    max_length: int = 200
    num_return_sequences: int = 4
    temperature_range: Tuple[float, float] = (0.5, 1.2)
    temperature_steps: int = 4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    fp16: bool = True
    
    seed: int = 42
    distributed: bool = True
    num_workers: int = 4
    
    log_level: str = "INFO"
    experiment_name: str = field(default_factory=lambda: f"tpo_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    def __post_init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.temperatures = np.linspace(
            self.temperature_range[0],
            self.temperature_range[1],
            self.temperature_steps
        )
        self._validate_config()
    
    def _validate_config(self) -> None:
        assert self.batch_size > 0, "Batch size must be positive"
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.max_length > 0, "Max length must be positive"
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'TrainingConfig':
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f)

# config/__init__.py
from .config import TrainingConfig

__all__ = ['TrainingConfig']