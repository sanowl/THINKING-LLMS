import pytest
from pathlib import Path
import yaml
from config import TrainingConfig

def test_training_config_validation():
    """Test configuration validation."""
    # Test valid configuration
    config = TrainingConfig(
        base_model_name="gpt2",
        judge_model_name="gpt2",
        batch_size=32,
        learning_rate=2e-5
    )
    assert config.batch_size == 32
    assert config.learning_rate == 2e-5

    # Test invalid batch size
    with pytest.raises(AssertionError):
        TrainingConfig(
            base_model_name="gpt2",
            judge_model_name="gpt2",
            batch_size=-1
        )

def test_config_yaml_loading(tmp_path):
    """Test loading configuration from YAML."""
    config_path = tmp_path / "test_config.yaml"
    config_data = {
        "base_model_name": "gpt2",
        "judge_model_name": "gpt2",
        "batch_size": 16,
        "learning_rate": 1e-5
    }
    
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    
    config = TrainingConfig.from_yaml(str(config_path))
    assert config.base_model_name == "gpt2"
    assert config.batch_size == 16
    assert config.learning_rate == 1e-5 