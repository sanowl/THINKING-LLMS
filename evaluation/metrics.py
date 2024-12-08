# evaluation/metrics.py
import numpy as np
from typing import List
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

class BaseMetric:
    """Base class for metrics."""
    
    def compute_score(self, prompt: str, text: str) -> float:
        """Compute metric score."""
        raise NotImplementedError

class ResponseQualityMetric(BaseMetric):
    """Metric for evaluating response quality."""
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    
    def compute_score(self, prompt: str, response: str) -> float:
        rouge_scores = self.rouge_scorer.score(prompt, response)
        
        # Combine different aspects of quality
        relevance = rouge_scores['rouge1'].fmeasure
        coherence = rouge_scores['rouge2'].fmeasure
        completeness = rouge_scores['rougeL'].fmeasure
        
        return (0.4 * relevance + 0.3 * coherence + 0.3 * completeness)

class ThoughtQualityMetric(BaseMetric):
    """Metric for evaluating thought process quality."""
    
    def __init__(self, sentence_model):
        self.sentence_model = sentence_model
    
    def compute_score(self, prompt: str, thought: str) -> float:
        # Split thought into components
        components = thought.split('.')
        if len(components) < 2:
            return 0.0
        
        # Calculate embeddings
        embeddings = self.sentence_model.encode([c.strip() for c in components if c.strip()])
        
        # Calculate coherence
        similarities = np.array([
            np.dot(embeddings[i], embeddings[i+1]) / 
            (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1]))
            for i in range(len(embeddings)-1)
        ])
        
        coherence = np.mean(similarities)
        
        # Calculate structure score
        keywords = ['consider', 'analyze', 'evaluate', 'conclude']
        structure_score = sum(
            1 for keyword in keywords if keyword in thought.lower()
        ) / len(keywords)
        
        return 0.7 * coherence + 0.3 * structure_score