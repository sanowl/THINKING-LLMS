import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import numpy as np
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import secrets

model_name: str = "meta-llama/Llama-3.2-1B"
model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_name)

judge_model_name: str = "meta-llama/Llama-3.2-70B"
judge_model = AutoModelForCausalLM.from_pretrained(judge_model_name, torch_dtype=torch.bfloat16, device_map="auto")
judge_tokenizer = AutoTokenizer.from_pretrained(judge_model_name)

sentence_model: SentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2')

def dpo_loss(chosen_logps: torch.Tensor, rejected_logps: torch.Tensor, beta: float = 0.1) -> torch.Tensor:
 return -torch.mean(torch.log(torch.sigmoid(beta * (chosen_logps - rejected_logps))))

def generate_thought_response(prompt: str, num_samples: int = 8, max_length: int = 200) -> List[Tuple[str, str]]:
 outputs: List[Tuple[str, str]] = []
 temperatures: List[float] = [0.5, 0.7, 0.9, 1.1]
 for temp in temperatures:
  for _ in range(num_samples // len(temperatures)):
   thought_prompt: str = f"Respond to the following user query in a comprehensive and detailed way. But first write down your internal thoughts. This must include your draft response and its evaluation. After this, write your final response after '<R>'.\n\nUser query: {prompt}"
   input_ids = tokenizer.encode(thought_prompt, return_tensors="pt").to(model.device)
   output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, do_sample=True, temperature=temp, top_p=0.95)
   decoded_output: str = tokenizer.decode(output[0], skip_special_tokens=True)
   parts: List[str] = decoded_output.split("<R>")
   thought: str = parts[0].strip()
   response: str = parts[1].strip() if len(parts) > 1 else ""
   outputs.append((thought, response))
 return outputs

def evaluate_responses(prompt: str, responses: List[str]) -> List[float]:
    scores = []
    for response in responses:
        input_text = f"Evaluate the response to the prompt: {prompt}\nResponse: {response}\nScore:"
        input_ids = judge_tokenizer.encode(input_text, return_tensors="pt").to(judge_model.device)
        with torch.no_grad():
            output = judge_model.generate(input_ids, max_length=50)
        score = judge_tokenizer.decode(output[0], skip_special_tokens=True)
        scores.append(float(score.strip().split()[-1])) 
    return scores

def evaluate_thought_quality(thoughts: List[str], prompt: str) -> List[float]:
 scores: List[float] = []
 prompt_embedding = sentence_model.encode([prompt])[0]
 for thought in thoughts:
  thought_parts: List[str] = thought.split('.')
  thought_embeddings = sentence_model.encode(thought_parts)
  coherence: float = np.mean([cosine_similarity([thought_embeddings[i]], [thought_embeddings[i+1]])[0][0] for i in range(len(thought_embeddings)-1)])
  thought_embedding = sentence_model.encode([thought])[0]
  relevance: float = cosine_similarity([prompt_embedding], [thought_embedding])[0][0]
  structure_score: float = (("draft" in thought.lower()) + ("evaluation" in thought.lower())) / 2
  scores.append(0.4 * coherence + 0.4 * relevance + 0.2 * structure_score)
 return scores

def apply_length_control(scores: List[float], responses: List[str], rho: float = 0.1) -> List[float]:
 lengths: List[int] = [len(r) for r in responses]
 mean_length: float = np.mean(lengths)
 std_length: float = np.std(lengths)
 normalized_scores: np.ndarray = (np.array(scores) - np.mean(scores)) / np.std(scores)
 normalized_lengths: np.ndarray = (np.array(lengths) - mean_length) / std_length
 controlled_scores: np.ndarray = normalized_scores - rho * normalized_lengths
 return controlled_scores.tolist()

def categorize_prompt(prompt: str) -> str:
 categories = ["Math", "Science", "History", "Literature", "General Knowledge"]
 # This needs to be better  but for now i will add this
 return secrets.choice(categories)

def train_tpo(num_iterations: int = 4, num_prompts: int = 5000, learning_rate: float = 1e-5) -> Dict[str, List[float]]:
 dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
 optimizer = AdamW(model.parameters(), lr=learning_rate)
 category_performances: Dict[str, List[float]] = {cat: [] for cat in ["Math", "Science", "History", "Literature", "General Knowledge"]}
 
 for iteration in tqdm(range(num_iterations), desc="Training Iterations"):
  prompts: List[str] = secrets.SystemRandom().sample(dataset['text'], num_prompts)
  for prompt in tqdm(prompts, desc="Prompts", leave=False):
   category = categorize_prompt(prompt)
   thought_responses: List[Tuple[str, str]] = generate_thought_response(prompt)
   thoughts, responses = zip(*thought_responses)
   response_scores: List[float] = evaluate_responses(prompt, responses)
   thought_scores: List[float] = evaluate_thought_quality(thoughts, prompt)
   combined_scores: List[float] = [0.6 * rs + 0.4 * ts for rs, ts in zip(response_scores, thought_scores)]
   controlled_scores: List[float] = apply_length_control(combined_scores, responses)
   best_index: int = np.argmax(controlled_scores)
   worst_index: int = np.argmin(controlled_scores)
   chosen_input = tokenizer(responses[best_index], return_tensors="pt").to(model.device)
   rejected_input = tokenizer(responses[worst_index], return_tensors="pt").to(model.device)
   with torch.no_grad():
    chosen_logps = model(**chosen_input).logits.log_softmax(-1)
    rejected_logps = model(**rejected_input).logits.log_softmax(-1)
   loss: torch.Tensor = dpo_loss(chosen_logps, rejected_logps)
   optimizer.zero_grad()
   loss.backward()
   optimizer.step()
   category_performances[category].append(controlled_scores[best_index])
  print(f"Completed iteration {iteration + 1}")
 model.save_pretrained("tpo_trained_model")
 return category_performances

def plot_category_performance(category_performances: Dict[str, List[float]]):
 plt.figure(figsize=(12, 6))
 for category, scores in category_performances.items():
  plt.plot(scores, label=category)
 plt.title("Performance by Category over Training")
 plt.xlabel("Training Steps")
 plt.ylabel("Score")
 plt.legend()
 plt.savefig("category_performance.png")
 plt.close()

def analyze_thought_lengths(thoughts: List[str]) -> None:
 lengths = [len(thought.split()) for thought in thoughts]
 plt.figure(figsize=(10, 5))
 plt.hist(lengths, bins=20)
 plt.title("Distribution of Thought Lengths")
 plt.xlabel("Number of Words")
 plt.ylabel("Frequency")
 plt.savefig("thought_length_distribution.png")
 plt.close()

category_performances = train_tpo(num_iterations=1, num_prompts=100)
plot_category_performance(category_performances)

user_prompt: str = "What is the meaning of life?"
thought_responses: List[Tuple[str, str]] = generate_thought_response(user_prompt, num_samples=10)
thoughts, responses = zip(*thought_responses)
analyze_thought_lengths(thoughts)

print("Thought Process:\n", thoughts[0])
print("\nFinal Response:\n", responses[0])
