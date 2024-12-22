import pytest
from transformers import AutoTokenizer
from data import ThoughtDataset

@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("gpt2")

def test_thought_dataset(tokenizer):
    """Test ThoughtDataset functionality."""
    prompts = [
        "What is machine learning?",
        "Explain neural networks.",
        "How does backpropagation work?"
    ]
    max_length = 128
    
    dataset = ThoughtDataset(prompts, tokenizer, max_length)
    
    # Test dataset length
    assert len(dataset) == len(prompts)
    
    # Test dataset item
    item = dataset[0]
    assert "input_ids" in item
    assert "attention_mask" in item
    assert item["input_ids"].shape[0] <= max_length
    assert item["attention_mask"].shape[0] <= max_length

def test_dataset_max_length(tokenizer):
    """Test dataset max length handling."""
    long_prompt = "This is a very long prompt. " * 100
    max_length = 50
    
    dataset = ThoughtDataset([long_prompt], tokenizer, max_length)
    item = dataset[0]
    
    assert item["input_ids"].shape[0] == max_length
    assert item["attention_mask"].shape[0] == max_length 