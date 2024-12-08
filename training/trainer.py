# training/trainer.py
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
from pathlib import Path
import wandb
from rich.progress import Progress, SpinnerColumn, TextColumn
from ..config import TrainingConfig
from ..models import ModelManager
from ..evaluation import Evaluator
from ..utils.logging import get_logger
from .optimizer import create_optimizer, create_scheduler

logger = get_logger(__name__)

class Trainer:
    """Handles the training loop and optimization."""
    
    def __init__(
        self,
        config: TrainingConfig,
        model_manager: ModelManager,
        evaluator: Evaluator
    ):
        self.config = config
        self.model_manager = model_manager
        self.evaluator = evaluator
        self.scaler = GradScaler() if config.fp16 else None
        
        # Initialize WandB for experiment tracking
        if self.model_manager.rank == 0:
            wandb.init(
                project="tpo",
                name=config.experiment_name,
                config=config.__dict__
            )
    
    def train(self, train_dataloader: DataLoader) -> None:
        """Execute training loop with comprehensive monitoring."""
        model = self.model_manager.model
        optimizer = create_optimizer(model, self.config)
        scheduler = create_scheduler(optimizer, len(train_dataloader), self.config)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            task = progress.add_task(
                "[cyan]Training...",
                total=self.config.max_steps
            )
            
            for step, batch in enumerate(train_dataloader):
                if step >= self.config.max_steps:
                    break
                
                loss = self._training_step(model, batch, optimizer, scheduler)
                
                # Evaluation and logging
                if step % self.config.eval_steps == 0:
                    metrics = self.evaluator.evaluate(model)
                    self._log_metrics(step, loss, metrics)
                
                # Save checkpoint
                if step % self.config.save_steps == 0:
                    self._save_checkpoint(model, optimizer, scheduler, step, metrics)
                
                progress.update(task, advance=1)
    
    def _training_step(
        self,
        model: torch.nn.Module,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        scheduler: Any
    ) -> float:
        """Execute single training step with gradient accumulation."""
        model.train()
        total_loss = 0
        
        for micro_step in range(self.config.gradient_accumulation_steps):
            with autocast(enabled=self.config.fp16):
                outputs = model(**batch)
                micro_loss = outputs.loss / self.config.gradient_accumulation_steps
            
            if self.scaler:
                self.scaler.scale(micro_loss).backward()
            else:
                micro_loss.backward()
            
            total_loss += micro_loss.item()
        
        # Optimizer step with gradient clipping
        if self.scaler:
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                self.config.max_grad_norm
            )
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                self.config.max_grad_norm
            )
            optimizer.step()
        
        scheduler.step()
        optimizer.zero_grad()
        
        return total_loss
    
    def _save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        step: int,
        metrics: Dict[str, float]
    ) -> None:
        """Save training checkpoint."""
        checkpoint_dir = Path("checkpoints") / self.config.experiment_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"checkpoint-{step}.pt"
        self.model_manager.save_checkpoint(
            model,
            optimizer,
            scheduler,
            step,
            metrics,
            str(checkpoint_path)
        )
    
    def _log_metrics(
        self,
        step: int,
        loss: float,
        metrics: Dict[str, float]
    ) -> None:
        """Log metrics to wandb and console."""
        if self.model_manager.rank == 0:
            metrics["loss"] = loss
            wandb.log(metrics, step=step)
            
            metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            logger.info(f"Step {step}: {metrics_str}")

