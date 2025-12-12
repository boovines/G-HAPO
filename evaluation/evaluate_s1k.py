import chz
import json
import logging
import re
import math
from typing import List, Dict, Any, Optional

import tinker
from tinker import types
import tinker_cookbook.model_info as model_info
import tinker_cookbook.renderers as renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tqdm import tqdm
import os

import sys
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory (project root)
parent_dir = os.path.dirname(current_dir)

# Add parent directory to sys.path
sys.path.insert(0, parent_dir)

from prompting_utils import SYSTEM_PROMPT
from math_utils import is_correct as math_utils_is_correct

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@chz.chz
class Config:
    max_tokens: int = 8192
    eval_temperature: float = 0.6
    eval_top_p: float = 0.95
    eval_batch_size: int = 8  # Increased batch size for parallel processing
    # Default to the s1k subset test file
    data_path: str = "../datasets/test_s1k.json" 

# ============================================================
# ANSWER EXTRACTION HELPERS
# ============================================================

def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract content from \boxed{...} handling nested braces."""
    if not text:
        return None
    
    # Find the last \boxed{ occurrence
    idx = text.rfind("\\boxed{")
    if idx < 0:
        # Try \boxed without brace
        if "\\boxed " in text:
            return text.split("\\boxed ")[-1].split("$")[0].strip()
        return None
    
    # Handle nested braces
    i = idx + 7  # len("\\boxed{")
    brace_count = 1
    while i < len(text) and brace_count > 0:
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
        i += 1
    
    if brace_count == 0:
        return text[idx + 7:i - 1].strip()
    return None

def extract_model_answer(response: str) -> str:
    """Extract the final answer from model response."""
    if not response:
        return ""
    
    # Try boxed answer first
    boxed = extract_boxed_answer(response)
    if boxed:
        return boxed
    
    # Try "The final answer is X" pattern
    match = re.search(r'(?i)the final answer is[:\s]*([^\n.]+)', response)
    if match:
        return match.group(1).strip()
    
    # Try "Answer: X" pattern  
    match = re.search(r'(?i)\banswer[:\s]+([^\n]+)', response)
    if match:
        return match.group(1).strip()
    
    # Return last non-empty line as fallback
    lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
    return lines[-1] if lines else ""

# ============================================================
# CROSSWORD EVALUATION
# ============================================================

def extract_crossword_answer(solution: str) -> str:
    """Extract answer from crossword solution format: '### Answer: WORD'"""
    match = re.search(r'###\s*Answer:\s*(\w+)', solution, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return solution.strip().upper()

def is_correct_crossword(response: str, ground_truth: str) -> bool:
    """Check if crossword answer matches (case-insensitive word match)."""
    model_answer = extract_model_answer(response).upper()
    gold_answer = extract_crossword_answer(ground_truth)
    
    # Clean up - remove punctuation and extra whitespace
    model_clean = re.sub(r'[^\w\s]', '', model_answer).strip()
    gold_clean = re.sub(r'[^\w\s]', '', gold_answer).strip()
    
    # Exact match
    if model_clean == gold_clean:
        return True
    
    # Check if gold answer is contained in model answer (for cases like "The answer is SOUTHBOUND")
    if gold_clean in model_clean:
        return True
    
    return False

# ============================================================
# SCIENCE (GPQA/JEE) EVALUATION  
# ============================================================

def extract_science_answer(solution: str, metadata: str = "") -> str:
    """Extract answer from science solution - handles multiple formats."""
    solution = solution.strip()
    
    # Format 1: Short answer like "ABC", "A", "10.02", "CO"
    if len(solution) < 20 and not solution.startswith("The") and not solution.startswith("For"):
        return solution
    
    # Format 2: Try to get from metadata (GPQA has 'Pre-Revision Correct Answer')
    if metadata:
        try:
            meta_dict = eval(metadata) if isinstance(metadata, str) else metadata
            if isinstance(meta_dict, dict):
                if 'Pre-Revision Correct Answer' in meta_dict:
                    return str(meta_dict['Pre-Revision Correct Answer'])
        except:
            pass
    
    # Format 3: Look for "correct answer is X" pattern
    match = re.search(r'(?i)correct answer is[:\s]*([^\n.,]+)', solution)
    if match:
        return match.group(1).strip()
    
    # Format 4: Look for answer letter pattern "(A)" at start or "answer: A"
    match = re.search(r'(?i)answer[:\s]*\(?([A-D])\)?', solution[:200])
    if match:
        return match.group(1).upper()
    
    # Fallback: return first line/sentence
    first_line = solution.split('\n')[0].split('.')[0].strip()
    return first_line[:100]

def normalize_science_answer(text: str) -> str:
    """Normalize science answer for comparison."""
    text = text.strip().upper()
    # Remove common prefixes
    text = re.sub(r'^(THE\s+ANSWER\s+IS|ANSWER|OPTION)[:\s]*', '', text, flags=re.IGNORECASE)
    # Remove parentheses around single letters
    text = re.sub(r'^\(([A-D])\)$', r'\1', text)
    return text.strip()

def is_correct_science(response: str, ground_truth: str, metadata: str = "") -> bool:
    """Check if science answer matches."""
    model_answer = extract_model_answer(response)
    gold_answer = extract_science_answer(ground_truth, metadata)
    
    model_norm = normalize_science_answer(model_answer)
    gold_norm = normalize_science_answer(gold_answer)
    
    # Direct match
    if model_norm == gold_norm:
        return True
    
    # Check if both are single letters (multiple choice)
    if len(gold_norm) <= 3 and gold_norm.isalpha():
        # Gold is likely multiple choice like "A", "B", "ABC"
        if model_norm == gold_norm:
            return True
        # Check if model answer contains the correct option
        if len(gold_norm) == 1 and gold_norm in model_norm[:5]:
            return True
    
    # Try numerical comparison for numeric answers
    try:
        model_num = float(re.sub(r'[^\d.\-]', '', model_norm))
        gold_num = float(re.sub(r'[^\d.\-]', '', gold_norm))
        if math.isclose(model_num, gold_num, rel_tol=0.01):
            return True
    except:
        pass
    
    # Substring match for longer answers
    if len(gold_norm) > 3 and gold_norm in model_norm:
        return True
    
    return False

# ============================================================
# MATH EVALUATION
# ============================================================

def is_correct_math(response: str, ground_truth: str) -> bool:
    """Check if math answer matches using math_verify library."""
    try:
        return math_utils_is_correct(response, ground_truth, use_math_verify=True)
    except Exception as e:
        logger.debug(f"math_verify failed, falling back to string comparison: {e}")
        # Fallback to simple comparison
        model_answer = extract_model_answer(response)
        gold_answer = ground_truth.strip()
        
        # Try to extract just the number if solution is a plain number
        if re.match(r'^-?\d+\.?\d*$', gold_answer):
            model_clean = re.sub(r'[^\d.\-]', '', model_answer)
            try:
                return math.isclose(float(model_clean), float(gold_answer), rel_tol=1e-5)
            except:
                pass
        
        return model_answer.strip().lower() == gold_answer.lower()

# ============================================================
# UNIFIED EVALUATION DISPATCHER
# ============================================================

def is_correct(response: str, ground_truth: str, cot_type: str, metadata: str = "") -> bool:
    """Dispatch to appropriate evaluation function based on question type."""
    cot_type = cot_type.lower() if cot_type else "math"
    
    if cot_type == "crossword":
        return is_correct_crossword(response, ground_truth)
    elif cot_type == "science":
        return is_correct_science(response, ground_truth, metadata)
    else:  # math or unknown
        return is_correct_math(response, ground_truth)

def format_s1k_question(example: Dict[str, Any]) -> str:
    """Formats the S1K question for the model."""
    # S1K questions are generally self-contained in the 'question' field
    return example["question"]

def evaluate(sampling_client, dataset, renderer, config):
    correct_count = 0
    total_count = 0
    total_token_length = 0
    
    # Track per-type accuracy
    type_stats = {}  # {cot_type: {"correct": 0, "total": 0}}
    
    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=config.eval_temperature,
        top_p=config.eval_top_p
    )
    
    # Batch processing for parallel requests
    n_batches = (len(dataset) + config.eval_batch_size - 1) // config.eval_batch_size
    
    pbar = tqdm(total=len(dataset), desc="Evaluating S1K")
    
    for i in range(n_batches):
        batch_start = i * config.eval_batch_size
        batch_end = min((i + 1) * config.eval_batch_size, len(dataset))
        batch = dataset[batch_start:batch_end]
        
        # Send all requests in parallel
        prompt_futures = []
        batch_metadata = []  # Store (ground_truth, cot_type, metadata) tuples
        
        for item in batch:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["q_str"]}
            ]
            model_input = renderer.build_generation_prompt(messages)
            
            prompt_futures.append(
                sampling_client.sample(
                    prompt=model_input,
                    num_samples=1,
                    sampling_params=sampling_params
                )
            )
            batch_metadata.append({
                "ground_truth": item["ground_truth"],
                "cot_type": item.get("cot_type", "math"),
                "metadata": item.get("metadata", "")
            })
        
        # Collect results
        for future, meta in zip(prompt_futures, batch_metadata):
            gt = meta["ground_truth"]
            cot_type = meta["cot_type"]
            metadata = meta["metadata"]
            
            # Initialize type stats if needed
            if cot_type not in type_stats:
                type_stats[cot_type] = {"correct": 0, "total": 0}
            
            try:
                result = future.result()
                tokens = result.sequences[0].tokens
                parsed, _ = renderer.parse_response(tokens)
                response_text = parsed["content"] if parsed and "content" in parsed else ""
                
                total_token_length += len(tokens)
                
                # Use unified is_correct with type dispatch
                if is_correct(response_text, gt, cot_type, metadata):
                    correct_count += 1
                    type_stats[cot_type]["correct"] += 1
                    
            except Exception as e:
                logger.error(f"Error processing {cot_type} example: {e}")
            
            type_stats[cot_type]["total"] += 1
            total_count += 1
        
        pbar.update(len(batch))
        pbar.set_postfix({'acc': f'{correct_count/total_count:.2%}' if total_count > 0 else 'N/A'})
    
    pbar.close()

    accuracy = correct_count / total_count if total_count > 0 else 0
    avg_len = total_token_length / total_count if total_count > 0 else 0
    
    # Print overall results
    logger.info(f"S1K Eval Results - Accuracy: {accuracy:.2%}, Avg Length: {avg_len:.1f}")
    print(f"\n=== S1K Eval Results ===")
    print(f"Overall Accuracy: {accuracy:.2%} ({correct_count}/{total_count})")
    print(f"Avg Token Length: {avg_len:.1f}")
    
    # Print per-type breakdown
    print(f"\n=== Per-Type Breakdown ===")
    for cot_type, stats in sorted(type_stats.items()):
        type_acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {cot_type:12s}: {type_acc:.2%} ({stats['correct']}/{stats['total']})")
    
    return accuracy, avg_len, correct_count, total_count, type_stats

if __name__ == "__main__":
    config = Config()
    
    # Load S1K Data
    # S1K subset is typically JSONL (lines=True) based on previous steps
    logger.info(f"Loading data from {config.data_path}")
    
    prepared_dataset = []
    
    # Handle loading based on file extension or assume JSONL
    import pandas as pd
    try:
        # Utilizing pandas for robust JSONL reading as established in prior context
        df = pd.read_json(config.data_path, lines=True)
        
        for _, row in df.iterrows():
            prepared_dataset.append({
                "q_str": format_s1k_question(row),
                "ground_truth": row["solution"],
                "cot_type": row.get("cot_type", "math"),
                "metadata": str(row.get("metadata", ""))
            })
    except ValueError:
        # Fallback for standard JSON list
        with open(config.data_path, 'r') as f:
            data = json.load(f)
            for item in data:
                prepared_dataset.append({
                    "q_str": format_s1k_question(item),
                    "ground_truth": item["solution"],
                    "cot_type": item.get("cot_type", "math"),
                    "metadata": str(item.get("metadata", ""))
                })

    # Log dataset composition
    type_counts = {}
    for item in prepared_dataset:
        ct = item.get("cot_type", "unknown")
        type_counts[ct] = type_counts.get(ct, 0) + 1
    
    logger.info(f"Prepared S1K dataset: {len(prepared_dataset)} examples")
    logger.info(f"Dataset composition: {type_counts}")
    
    # Model setup
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    
    # Initialize tokenizer and renderer
    tokenizer = get_tokenizer(model_name)
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")
    
    # Create sampling client
    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(base_model=model_name, rank=8)
    
    # Qwen3-4B-Instruct tuned on original HAPO dataset
    # training_client.load_state("tinker://d88742cc-b842-58bd-9c6e-16281f28b3a0:train:0/weights/000120")

    # Qwen3-4B-Instruct tuning on mixed dataset
    training_client.load_state("tinker://8cd9fd36-35ab-530f-9226-e9be0f396858:train:0/weights/000120")

    sampling_client = training_client.save_weights_and_get_sampling_client(name="HAPO")
    logger.info(f"Created sampling client for HAPO model")
    
    # Run evaluation
    pass_at_1, average_token_length, correct_count, total_count, type_stats = evaluate(
        sampling_client, prepared_dataset, renderer, config
    )
    
    logger.info(f"Final Results: {correct_count}/{total_count} ({pass_at_1:.2%})")
    print(f"\n=== Final Results ===")
    print(f"Pass@1: {pass_at_1:.2%} ({correct_count}/{total_count})")
    
    # Save results to file for future reference
    import datetime
    
    # Build per-type accuracy dict for JSON
    per_type_results = {}
    for cot_type, stats in type_stats.items():
        per_type_results[cot_type] = {
            "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0,
            "correct": stats["correct"],
            "total": stats["total"]
        }
    
    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "dataset": "S1K",
        "data_path": config.data_path,
        "accuracy": pass_at_1,
        "correct": correct_count,
        "total": total_count,
        "avg_token_length": average_token_length,
        "per_type_results": per_type_results
    }
    results_file = "s1k_eval_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")