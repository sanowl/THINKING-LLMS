import pytest
import torch
from metrics import MetricsTracker, ResponseQualityMetric, ThoughtQualityMetric
from sentence_transformers import SentenceTransformer

@pytest.fixture
def metrics_tracker():
    return MetricsTracker()

@pytest.fixture
def response_metric():
    return ResponseQualityMetric()

@pytest.fixture
def thought_metric():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return ThoughtQualityMetric(model)

def test_metrics_tracker(metrics_tracker):
    """Test metrics tracking functionality."""
    metric = ResponseQualityMetric()
    metrics_tracker.add_metric("response_quality", metric)
    
    # Test updating metrics
    preds = torch.tensor([0.8, 0.6, 0.7])
    targets = torch.tensor([1.0, 0.5, 0.8])
    metrics_tracker.update("response_quality", preds, targets)
    
    # Test computing metrics
    results = metrics_tracker.compute_all()
    assert "response_quality" in results
    assert isinstance(results["response_quality"], float)

def test_response_quality_metric(response_metric):
    """Test response quality metric computation."""
    preds = torch.tensor([0.8, 0.6, 0.7])
    targets = torch.tensor([1.0, 0.5, 0.8])
    
    response_metric.update(preds, targets)
    score = response_metric.compute()
    
    assert isinstance(score, float)
    assert 0 <= score <= 1

def test_thought_quality_metric(thought_metric):
    """Test thought quality metric computation."""
    preds = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    targets = torch.tensor([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]])
    
    thought_metric.update(preds, targets)
    score = thought_metric.compute()
    
    assert isinstance(score, float)
    assert 0 <= score <= 1 