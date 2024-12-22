import pytest
import torch
from config import TrainingConfig

@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture(scope="session")
def base_config():
    return TrainingConfig(
        base_model_name="gpt2",
        judge_model_name="gpt2",
        distributed=False,
        batch_size=4,
        max_steps=10,
        eval_steps=2,
        save_steps=5
    )

@pytest.fixture(autouse=True)
def setup_teardown():
    # Setup before each test
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    yield
    
    # Teardown after each test
    torch.cuda.empty_cache() if torch.cuda.is_available() else None 