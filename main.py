# main.py
import os
import torch
from pathlib import Path
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from config import TrainingConfig
from models import ModelManager
from data import ThoughtDataset, create_dataloader
from training import Trainer
from evaluation import Evaluator
from utils import get_logger

logger = get_logger(__name__)

def setup_environment(config: TrainingConfig) -> None:
    """Setup training environment."""
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    os.makedirs("checkpoints", exist_ok=True)

def main():
    """Main execution function."""
    # Load configuration
    config = TrainingConfig.from_yaml("config.yaml")
    setup_environment(config)
    
    # Initialize components
    model_manager = ModelManager(config)
    model = model_manager.load_model(config.base_model_name)
    tokenizer = model_manager.load_tokenizer(config.base_model_name)
    sentence_model = SentenceTransformer(config.sentence_model_name)
    
    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    train_dataset = ThoughtDataset(
        dataset["text"],
        tokenizer,
        config.max_length
    )
    train_dataloader = create_dataloader(
        train_dataset,
        config.batch_size,
        config.num_workers
    )
    
    # Setup training components
    evaluator = Evaluator(config, sentence_model)
    trainer = Trainer(config, model_manager, evaluator)
    
    # Train model
    logger.info("Starting training...")
    trainer.train(train_dataloader)
    logger.info("Training completed!")
    
    # Generate example responses
    prompt = "Explain the concept of machine learning to a beginner."
    responses = evaluator._generate_responses(model, prompt)
    
    # Print example response
    thought, response = responses[0]
    logger.info(f"Example thought process:\n{thought}")
    logger.info(f"Example response:\n{response}")

if __name__ == "__main__":
    main()