

import os
import json
import random
import numpy as np
import torch
import torch.distributed as dist
import wandb
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Union, Any, Callable
from pathlib import Path
from datetime import datetime
from abc import ABC, abstractmethod
import logging
import yaml
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup
)
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
from typing_extensions import Protocol

# Setup enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("tpo")

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
    
    # Generation parameters
    max_length: int = 200
    num_return_sequences: int = 4
    temperature_range: Tuple[float, float] = (0.5, 1.2)
    temperature_steps: int = 4
    
    # Optimization parameters
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    fp16: bool = True
    
    # System parameters
    seed: int = 42
    distributed: bool = True
    num_workers: int = 4
    
    # Logging parameters
    log_level: str = "INFO"
    experiment_name: str = f"tpo_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def __post_init__(self):
        """Initialize derived parameters and validate configuration."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.temperatures = np.linspace(
            self.temperature_range[0],
            self.temperature_range[1],
            self.temperature_steps
        )
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        assert self.batch_size > 0, "Batch size must be positive"
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.max_length > 0, "Max length must be positive"
        
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'TrainingConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def save(self, path: str):
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f)

class Metric(Protocol):
    """Protocol for implementing custom metrics."""
    
    @abstractmethod
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """Update metric with new predictions and targets."""
        pass
    
    @abstractmethod
    def compute(self) -> float:
        """Compute and return the metric value."""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset metric state."""
        pass

class ThoughtDataset(Dataset):
    """Custom dataset for thought-response pairs."""
    
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

class ModelManager:
    """Manages model loading, saving, and initialization."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self._setup_environment()
    
    def _setup_environment(self):
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
    
    def save_checkpoint(
        self,
        model: PreTrainedModel,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        step: int,
        metrics: Dict[str, float],
        path: str
    ):
        """Save training checkpoint with all necessary information."""
        if self.rank == 0:
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

class MetricsTracker:
    """Tracks and logs multiple training metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, Metric] = {}
        self.history: Dict[str, List[float]] = {}
    
    def add_metric(self, name: str, metric: Metric):
        """Add a new metric to track."""
        self.metrics[name] = metric
        self.history[name] = []
    
    def update(self, name: str, preds: torch.Tensor, targets: torch.Tensor):
        """Update specific metric with new data."""
        self.metrics[name].update(preds, targets)
    
    def compute_all(self) -> Dict[str, float]:
        """Compute all metrics and store in history."""
        results = {}
        for name, metric in self.metrics.items():
            value = metric.compute()
            results[name] = value
            self.history[name].append(value)
            metric.reset()
        return results
    
    def plot_history(self, save_path: Optional[str] = None):
        """Plot metric history."""
        plt.figure(figsize=(12, 6))
        for name, values in self.history.items():
            plt.plot(values, label=name)
        plt.title("Training Metrics History")
        plt.xlabel("Steps")
        plt.ylabel("Value")
        plt.legend()
        if save_path:
            plt.savefig(save_path)
        plt.close()

class AdvancedTPO:
    """Advanced Thought Process Optimization implementation."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model_manager = ModelManager(config)
        self.metrics_tracker = MetricsTracker()
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize all model components and utilities."""
        self.model = self.model_manager.load_model(self.config.base_model_name)
        self.judge_model = self.model_manager.load_model(self.config.judge_model_name)
        self.sentence_model = SentenceTransformer(self.config.sentence_model_name)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name)
        self.judge_tokenizer = AutoTokenizer.from_pretrained(self.config.judge_model_name)
        
        self.scaler = GradScaler() if self.config.fp16 else None
        
        # Initialize WandB for experiment tracking
        if self.model_manager.rank == 0:
            wandb.init(
                project="tpo",
                name=self.config.experiment_name,
                config=self.config.__dict__
            )
    
    def train(self, dataset: Dataset):
        """Execute advanced training loop with comprehensive monitoring."""
        optimizer = self._create_optimizer()
        scheduler = self._create_scheduler(optimizer, len(dataset))
        
        train_dataloader = self._create_dataloader(dataset)
        
        # Training loop with progress tracking
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
                
                loss = self._training_step(batch, optimizer, scheduler)
                
                # Log metrics and save checkpoints
                if step % self.config.eval_steps == 0:
                    metrics = self._evaluate()
                    self._log_metrics(step, loss, metrics)
                
                if step % self.config.save_steps == 0:
                    self._save_checkpoint(step, metrics)
                
                progress.update(task, advance=1)
    
    def _training_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        scheduler: Any
    ) -> float:
        """Execute single training step with gradient accumulation."""
        self.model.train()
        
        loss = 0
        for micro_step in range(self.config.gradient_accumulation_steps):
            with autocast(enabled=self.config.fp16):
                outputs = self._forward_pass(batch)
                micro_loss = outputs.loss / self.config.gradient_accumulation_steps
            
            if self.scaler:
                self.scaler.scale(micro_loss).backward()
            else:
                micro_loss.backward()
            
            loss += micro_loss.item()
        
        # Optimizer step with gradient clipping
        if self.scaler:
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            optimizer.step()
        
        scheduler.step()
        optimizer.zero_grad()
        
        return loss
    
    def _evaluate(self) -> Dict[str, float]:
        """Comprehensive model evaluation."""
        self.model.eval()
        metrics = {}
        
        # Generate and evaluate samples
        with torch.no_grad():
            eval_prompts = self._get_eval_prompts()
            for prompt in eval_prompts:
                responses = self.generate_responses(prompt)
                metrics.update(self._calculate_metrics(prompt, responses))
        
        return metrics
    
    def generate_responses(
        self,
        prompt: str,
        num_samples: Optional[int] = None
    ) -> List[Tuple[str, str]]:
        """Generate multiple responses with different temperatures."""
        if num_samples is None:
            num_samples = self.config.num_return_sequences
        
        responses = []
        for temp in self.config.temperatures:
            for _ in range(num_samples // len(self.config.temperatures)):
                response = self._generate_single_response(prompt, temp)
                responses.append(response)
        
        return responses
    
    def _generate_single_response(
        self,
        prompt: str,
        temperature: float
    ) -> Tuple[str, str]:
        """Generate a single response with thought process."""
        formatted_prompt = self._format_prompt(prompt)
        input_ids = self.tokenizer(
            formatted_prompt,
            return_tensors="pt"
        ).input_ids.to(self.config.device)
        
        with autocast(enabled=self.config.fp16):
            output_ids = self.model.generate(
                input_ids,
                max_length=self.config.max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                num_return_sequences=1
            )
        
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        thought, response = self._split_output(output_text)
        
        return thought, response
    
    def _calculate_metrics(
        self,
        prompt: str,
        responses: List[Tuple[str, str]]
    ) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        metrics = {}
        
        # Response quality metrics
        response_scores = self._evaluate_responses(prompt, [r[1] for r in responses])
        metrics["response_quality"] = np.mean(response_scores)
        
        # Thought process metrics
        thought_scores = self._evaluate_thoughts(prompt, [r[0] for r in responses])
        metrics["thought_quality"]

class AdvancedMetric(Metric):
    """Implementation of advanced evaluation metrics."""
    
    def __init__(self):
        self.reset()
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        self.predictions.extend(preds.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
    
    def compute(self) -> float:
        if not self.predictions:
            return 0.0
        return np.mean(self.predictions)
    
    def reset(self):
        self.predictions = []
        self.targets = []

class ResponseQualityMetric(AdvancedMetric):
    """Metric for evaluating response quality."""
    
    def compute(self) -> float:
        if not self.predictions:
            return 0.0
        
        # Calculate BLEU, ROUGE, and coherence scores
        bleu_scores = [
            sentence_bleu([ref], hyp) 
            for ref, hyp in zip(self.targets, self.predictions)
        ]
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
        rouge_scores = [
            scorer.score(ref, hyp)
            for ref, hyp in zip(self.targets, self.predictions)
        ]
        
        # Combine scores with weights
        final_scores = [
            0.4 * bleu + 0.3 * rouge['rouge1'].fmeasure + 
            0.3 * rouge['rouge2'].fmeasure
            for bleu, rouge in zip(bleu_scores, rouge_scores)
        ]
        
        return np.mean(final_scores)

class ThoughtQualityMetric(AdvancedMetric):
    """Metric for evaluating thought process quality."""
    
    def __init__(self, sentence_model: SentenceTransformer):
        super().__init__()
        self.sentence_model = sentence_model
    
    def compute(self) -> float:
        if not self.predictions:
            return 0.0
        
        # Evaluate thought coherence and structure
        coherence_scores = []
        for thought in self.predictions:
            # Split thought into sentences
            sentences = thought.split('.')
            if len(sentences) < 2:
                coherence_scores.append(0.0)
                continue
            
            # Calculate embeddings and coherence
            embeddings = self.sentence_model.encode(sentences)
            pairwise_similarities = cosine_similarity(embeddings)
            coherence = np.mean(
                [pairwise_similarities[i][i+1] 
                 for i in range(len(sentences)-1)]
            )
            coherence_scores.append(coherence)
        
        return np.mean(coherence_scores)

class AdvancedTPO:
    """Continuing the AdvancedTPO implementation..."""
    
    def _calculate_metrics(
        self,
        prompt: str,
        responses: List[Tuple[str, str]]
    ) -> Dict[str, float]:
        metrics = {}
        
        # Response quality metrics
        response_scores = self._evaluate_responses(prompt, [r[1] for r in responses])
        metrics["response_quality"] = np.mean(response_scores)
        
        # Thought process metrics
        thought_scores = self._evaluate_thoughts(prompt, [r[0] for r in responses])
        metrics["thought_quality"] = np.mean(thought_scores)
        
        # Diversity metrics
        metrics["response_diversity"] = self._calculate_diversity([r[1] for r in responses])
        metrics["thought_diversity"] = self._calculate_diversity([r[0] for r in responses])
        
        # Length and structure metrics
        metrics.update(self._calculate_structural_metrics(responses))
        
        return metrics
    
    def _evaluate_responses(self, prompt: str, responses: List[str]) -> List[float]:
        """Evaluate response quality using the judge model."""
        scores = []
        for response in responses:
            # Prepare input for judge model
            input_text = (
                f"Rate the following response (0-10) for clarity, relevance, "
                f"and completeness:\nPrompt: {prompt}\nResponse: {response}\nRating:"
            )
            
            with torch.no_grad(), autocast(enabled=self.config.fp16):
                inputs = self.judge_tokenizer(
                    input_text,
                    return_tensors="pt",
                    max_length=self.config.max_length,
                    truncation=True
                ).to(self.config.device)
                
                outputs = self.judge_model.generate(
                    **inputs,
                    max_length=50,
                    num_return_sequences=1,
                    temperature=0.7
                )
                
                score_text = self.judge_tokenizer.decode(outputs[0], skip_special_tokens=True)
                try:
                    score = float(score_text.strip().split()[-1])
                    scores.append(min(max(score, 0), 10) / 10)
                except ValueError:
                    scores.append(0.5)  # Default score if parsing fails
        
        return scores
    
    def _evaluate_thoughts(self, prompt: str, thoughts: List[str]) -> List[float]:
        """Evaluate thought process quality."""
        scores = []
        prompt_embedding = self.sentence_model.encode([prompt])[0]
        
        for thought in thoughts:
            # Split thought into components
            components = thought.lower().split('\n')
            
            # Calculate coherence
            thought_embeddings = self.sentence_model.encode(components)
            coherence = np.mean([
                cosine_similarity([thought_embeddings[i]], [thought_embeddings[i+1]])[0][0]
                for i in range(len(thought_embeddings)-1)
            ])
            
            # Calculate relevance to prompt
            thought_embedding = np.mean(thought_embeddings, axis=0)
            relevance = cosine_similarity([prompt_embedding], [thought_embedding])[0][0]
            
            # Calculate structure score
            structure_score = sum(
                1 for keyword in ['consider', 'analyze', 'evaluate', 'conclude']
                if keyword in thought.lower()
            ) / 4
            
            # Combine scores
            final_score = 0.4 * coherence + 0.4 * relevance + 0.2 * structure_score
            scores.append(final_score)
        
        return scores
    
    def _calculate_diversity(self, texts: List[str]) -> float:
        """Calculate diversity score among generated texts."""
        if len(texts) < 2:
            return 0.0
        
        # Calculate embeddings for all texts
        embeddings = self.sentence_model.encode(texts)
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(embeddings)
        
        # Calculate diversity as 1 - average similarity
        diversity = 1 - (np.sum(similarities) - len(texts)) / (len(texts) * (len(texts) - 1))
        
        return diversity
    
    def _calculate_structural_metrics(
        self,
        responses: List[Tuple[str, str]]
    ) -> Dict[str, float]:
        """Calculate structural metrics for thoughts and responses."""
        metrics = {}
        
        # Length metrics
        thought_lengths = [len(t.split()) for t, _ in responses]
        response_lengths = [len(r.split()) for _, r in responses]
        
        metrics["avg_thought_length"] = np.mean(thought_lengths)
        metrics["avg_response_length"] = np.mean(response_lengths)
        
        # Complexity metrics
        metrics["thought_complexity"] = self._calculate_complexity([t for t, _ in responses])
        metrics["response_complexity"] = self._calculate_complexity([r for _, r in responses])
        
        return metrics
    
    def _calculate_complexity(self, texts: List[str]) -> float:
        """Calculate linguistic complexity score."""
        if not texts:
            return 0.0
        
        def get_complexity(text: str) -> float:
            sentences = text.split('.')
            word_counts = [len(s.split()) for s in sentences if s.strip()]
            if not word_counts:
                return 0.0
            
            # Combine average sentence length and vocabulary diversity
            avg_length = np.mean(word_counts)
            unique_words = len(set(text.lower().split()))
            total_words = len(text.split())
            
            return 0.5 * (avg_length / 20) + 0.5 * (unique_words / total_words)
        
        return np.mean([get_complexity(text) for text in texts])
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with weight decay."""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters()
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate)
    
    def _create_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        num_training_steps: int
    ) -> Any:
        """Create learning rate scheduler."""
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps
        )
    
    def _create_dataloader(self, dataset: Dataset) -> DataLoader:
        """Create training dataloader with proper settings."""
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    def _save_checkpoint(self, step: int, metrics: Dict[str, float]):
        """Save training checkpoint."""
        checkpoint_dir = Path("checkpoints") / self.config.experiment_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"checkpoint-{step}.pt"
        self.model_manager.save_checkpoint(
            self.model,
            self.optimizer,
            self.scheduler,
            step,
            metrics,
            str(checkpoint_path)
        )
        
        # Save configuration
        config_path = checkpoint_dir / "config.yaml"
        self.config.save(str(config_path))
    
    def _log_metrics(self, step: int, loss: float, metrics: Dict[str, float]):
        """Log metrics to wandb and console."""
        if self.model_manager.rank == 0:
            metrics["loss"] = loss
            wandb.log(metrics, step=step)
            
            # Log to console
            metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            logger.info(f"Step {step}: {metrics_str}")

def main():
    """Main execution function."""
    # Load configuration
    config = TrainingConfig.from_yaml("config.yaml")
    
    # Initialize TPO
    tpo = AdvancedTPO(config)
    
    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    # Create training dataset
    train_dataset = ThoughtDataset(
        dataset["text"],
        tpo.tokenizer,
        config.max_length
    )
    
    # Train model
    tpo.train(train_dataset)
    
    # Generate example responses
    prompt = "Explain the concept of machine learning to a beginner."
    responses = tpo.generate_responses(prompt)
    
    # Print example response
    thought, response = responses[0]
    logger.info(f"Example thought process:\n{thought}")
    logger.info(f"Example response:\n{response}")

if __name__ == "__main__":
    main()