# evaluation/evaluator.py
import torch
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from ..config import TrainingConfig
from ..utils.logging import get_logger
from .metrics import ResponseQualityMetric, ThoughtQualityMetric

logger = get_logger(__name__)

class Evaluator:
    """Handles model evaluation and metrics calculation."""
    
    def __init__(self, config: TrainingConfig, sentence_model):
        self.config = config
        self.sentence_model = sentence_model
        self.response_metric = ResponseQualityMetric()
        self.thought_metric = ThoughtQualityMetric(sentence_model)
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    
    def evaluate(self, model: torch.nn.Module) -> Dict[str, float]:
        """Comprehensive model evaluation."""
        model.eval()
        metrics = {}
        
        eval_prompts = self._get_eval_prompts()
        for prompt in eval_prompts:
            responses = self._generate_responses(model, prompt)
            prompt_metrics = self._calculate_metrics(prompt, responses)
            for k, v in prompt_metrics.items():
                metrics[k] = metrics.get(k, 0.0) + v / len(eval_prompts)
        
        return metrics
    
    def _generate_responses(
        self,
        model: torch.nn.Module,
        prompt: str,
        num_samples: int = None
    ) -> List[Tuple[str, str]]:
        """Generate multiple responses with different temperatures."""
        if num_samples is None:
            num_samples = self.config.num_return_sequences
        
        responses = []
        with torch.no_grad():
            for temp in self.config.temperatures:
                for _ in range(num_samples // len(self.config.temperatures)):
                    thought, response = self._generate_single_response(
                        model, prompt, temp
                    )
                    responses.append((thought, response))
        
        return responses
    
    def _generate_single_response(
        self,
        model: torch.nn.Module,
        prompt: str,
        temperature: float
    ) -> Tuple[str, str]:
        """Generate a single response with thought process."""
        input_text = self._format_prompt(prompt)
        input_ids = self.tokenizer(
            input_text,
            return_tensors="pt"
        ).input_ids.to(self.config.device)
        
        with torch.cuda.amp.autocast(enabled=self.config.fp16):
            output_ids = model.generate(
                input_ids,
                max_length=self.config.max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                num_return_sequences=1
            )
        
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return self._split_output(output_text)
    
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
        metrics["thought_quality"] = np.mean(thought_scores)
        
        # Diversity metrics
        metrics["response_diversity"] = self._calculate_diversity(
            [r[1] for r in responses]
        )
        metrics["thought_diversity"] = self._calculate_diversity(
            [r[0] for r in responses]
        )
        
        # Additional metrics
        metrics.update(self._calculate_structural_metrics(responses))
        
        return metrics
    
    def _evaluate_responses(self, prompt: str, responses: List[str]) -> List[float]:
        """Evaluate response quality."""
        return [self.response_metric.compute_score(prompt, response) 
                for response in responses]
    
    def _evaluate_thoughts(self, prompt: str, thoughts: List[str]) -> List[float]:
        """Evaluate thought process quality."""
        return [self.thought_metric.compute_score(prompt, thought) 
                for thought in thoughts]
    
    def _calculate_diversity(self, texts: List[str]) -> float:
        """Calculate diversity score among generated texts."""
        if len(texts) < 2:
            return 0.0
        
        embeddings = self.sentence_model.encode(texts)
        similarities = cosine_similarity(embeddings)
        
        return 1 - (np.sum(similarities) - len(texts)) / (len(texts) * (len(texts) - 1))
    
    def _calculate_structural_metrics(
        self,
        responses: List[Tuple[str, str]]
    ) -> Dict[str, float]:
        """Calculate structural metrics for thoughts and responses."""
        metrics = {}
        
        thought_lengths = [len(t.split()) for t, _ in responses]
        response_lengths = [len(r.split()) for _, r in responses]
        
        metrics["avg_thought_length"] = np.mean(thought_lengths)
        metrics["avg_response_length"] = np.mean(response_lengths)
        
        return metrics
    
    def _format_prompt(self, prompt: str) -> str:
        """Format prompt for model input."""
        return f"Question: {prompt}\nThought:"
    
    def _split_output(self, output_text: str) -> Tuple[str, str]:
        """Split output into thought and response."""
        parts = output_text.split("\nResponse:")
        if len(parts) != 2:
            return output_text, ""
        return parts[0].strip(), parts[1].strip()
    
    def _get_eval_prompts(self) -> List[str]:
        """Get evaluation prompts."""
        # get the real files later
        return [
            "Explain the concept of machine learning.",
            "What are the main causes of climate change?",
            "How does photosynthesis work?",
        ]



