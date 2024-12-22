import pytest
import torch
from models import ModelManager
from config import TrainingConfig

@pytest.fixture
def config():
    return TrainingConfig(
        base_model_name="gpt2",
        judge_model_name="gpt2",
        distributed=False
    )

def test_model_loading(config):
    """Test model loading functionality."""
    manager = ModelManager(config)
    model = manager.load_model("gpt2")
    
    assert isinstance(model, torch.nn.Module)
    assert next(model.parameters()).device == config.device

def test_checkpoint_saving(config, tmp_path):
    """Test checkpoint saving functionality."""
    manager = ModelManager(config)
    model = manager.load_model("gpt2")
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
    
    checkpoint_path = tmp_path / "checkpoint.pt"
    metrics = {"loss": 0.5, "accuracy": 0.8}
    
    manager.save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=100,
        metrics=metrics,
        path=str(checkpoint_path)
    )
    
    assert checkpoint_path.exists()
    checkpoint = torch.load(str(checkpoint_path))
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert "metrics" in checkpoint
    assert checkpoint["step"] == 100
    assert checkpoint["metrics"] == metrics 