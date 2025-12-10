import chz
import json
import logging
import random
from typing import List, Dict, Any, Optional

import tinker
from tinker import types
import tinker_cookbook.model_info as model_info
import tinker_cookbook.renderers as renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tqdm import tqdm

import sys
sys.path.insert(0, '..')
from prompting_utils import SYSTEM_PROMPT

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@chz.chz
class Config:
    max_tokens: int = 8192
    eval_temperature: float = 0.6
    eval_top_p: float = 0.95
    eval_batch_size: int = 4
    data_path: str = "../datasets/test_reclor.json" #"../datasets/ReClor/val.json"

def is_correct_textual(response: str, ground_truth: str) -> bool:
    """
    Checks if the model's response matches the ground truth for textual multiple choice.
    
    Args:
        response: The model's generated text, potentially containing LaTeX boxed answer.
        ground_truth: The correct answer (e.g., 'A', 'B', 'C', 'D').
        
    Returns:
        True if the response contains the correct answer.
    """
    # Normalize ground truth
    gt = ground_truth.strip().upper()
    
    # Check for boxed answer first as per system prompt
    import re
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', response)
    if boxed_match:
        prediction = boxed_match.group(1).strip().upper()
        # Clean up prediction (e.g. if it is "Option A" or "A)")
        prediction = re.sub(r'^(OPTION|ANSWER)\s*', '', prediction)
        prediction = prediction.strip("()[]")
        if prediction == gt:
            return True
            
    # Fallback: Check if the last sentence or line contains the answer
    # This is a simple heuristic; might need refinement
    lines = response.strip().split('\n')
    if lines:
        last_line = lines[-1].upper()
        # Look for "Answer: A" or similar
        match = re.search(r'(?:ANSWER|OPTION)?\s*[:=]?\s*([A-D])\b', last_line)
        if match:
             if match.group(1) == gt:
                 return True

    # If the response strictly equals the ground truth (rare with chain of thought)
    if response.strip().upper() == gt:
        return True
        
    return False

def evaluate(client, dataset: List[Dict], renderer, config: Config):
    logger.info("Running validation evaluation...")
    logger.info(f"Dataset size: {len(dataset)} examples")
    logger.info(f"Batch size: {config.eval_batch_size}")
    logger.info(f"Temperature: {config.eval_temperature}, Top-p: {config.eval_top_p}")
    
    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=config.eval_temperature,
        top_p=config.eval_top_p
    )
    
    correct_count = 0
    total_count = 0
    token_lengths = []
    
    n_batches = (len(dataset) + config.eval_batch_size - 1) // config.eval_batch_size
    
    # Create progress bar
    pbar = tqdm(total=len(dataset), desc="Evaluating", unit="examples")
    
    for i in range(n_batches):
        batch = dataset[i * config.eval_batch_size : min((i + 1) * config.eval_batch_size, len(dataset))]
        
        batch_size = len(batch)
        prompt_futures = []
        ground_truths = [item["ground_truth"] for item in batch]
        
        for item in batch:
            q_str = item["q_str"]
            convo = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q_str}
            ]
            model_input = renderer.build_generation_prompt(convo)
            
            prompt_futures.append(
                client.sample(
                    prompt=model_input,
                    num_samples=1,
                    sampling_params=sampling_params
                )
            )
        
        batch_correct = 0
        for idx, (future, gt) in enumerate(zip(prompt_futures, ground_truths)):
            try:
                result = future.result()
                tokens = result.sequences[0].tokens
                parsed, _ = renderer.parse_response(tokens)
                content = parsed["content"] if parsed and "content" in parsed else ""

                if idx % 5 == 0:
                    print(f"Sample {idx} Response: {content}")
                
                if is_correct_textual(content, gt):
                    correct_count += 1
                    batch_correct += 1

                token_lengths.append(len(tokens))
            except Exception as e:
                logger.error(f"Error during eval sample: {e}")
                
            total_count += 1
        
        # Update progress bar with current stats
        current_accuracy = correct_count / total_count if total_count > 0 else 0.0
        avg_tokens = sum(token_lengths) / len(token_lengths) if token_lengths else 0.0
        pbar.set_postfix({
            'acc': f'{current_accuracy:.4f}',
            'batch_acc': f'{batch_correct}/{batch_size}',
            'avg_tokens': f'{avg_tokens:.0f}'
        })
        pbar.update(batch_size)
        
        # Log every 50 batches
        if (i + 1) % 50 == 0:
            logger.info(f"Progress: {total_count}/{len(dataset)} | Accuracy: {current_accuracy:.4f} | Avg tokens: {avg_tokens:.1f}")
    
    pbar.close()
            
    pass_at_1 = correct_count / total_count if total_count > 0 else 0.0
    average_token_length = sum(token_lengths) / len(token_lengths) if token_lengths else 0.0
    return pass_at_1, average_token_length, correct_count, total_count


def format_reclor_question(example: Dict[str, Any]) -> str:
    """Formats a ReClor example into a prompt string."""
    context = example.get("context", "")
    question = example.get("question", "")
    answers = example.get("answers", [])
    
    options_str = ""
    labels = ['A', 'B', 'C', 'D']
    for label, answer in zip(labels, answers):
        options_str += f"{label}) {answer}\n"
    
    return f"{context}\n\nQuestion: {question}\n\nOptions:\n{options_str}\nProvide the correct option letter."

def main(config: Config):
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Load ReClor test dataset (handles both JSON and JSONL formats)
    try:
        with open(config.data_path, 'r') as f:
            content = f.read().strip()
            # Try standard JSON first
            try:
                raw_data = json.loads(content)
            except json.JSONDecodeError:
                # Fall back to JSONL (one JSON object per line)
                raw_data = [json.loads(line) for line in content.split('\n') if line.strip()]
    except FileNotFoundError:
        logger.error(f"Could not find dataset at {config.data_path}")
        return

    # Randomly sample 50 examples
    # Ensure reproducibility
    random.seed(42)
    if len(raw_data) > 50:
        dataset = random.sample(raw_data, 50)
        logger.info(f"Randomly sampled 50 examples from ReClor test set")
    else:
        dataset = raw_data
        logger.info(f"Loaded full dataset ({len(dataset)} examples) as it is smaller than 50")
    
    # Prepare dataset format
    prepared_dataset = []
    for example in dataset:            
        label_idx = example["label"]
        ground_truth = chr(ord('A') + label_idx)

        prepared_dataset.append({
            "q_str": format_reclor_question(example),
            "ground_truth": ground_truth
        })
    
    logger.info(f"Prepared ReClor dataset: {len(prepared_dataset)} examples")
    
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
    # training_client.load_state("tinker://d88742cc-b842-58bd-9c6e-16281f28b3a0:train:0/weights/000120")
    sampling_client = training_client.save_weights_and_get_sampling_client(name="HAPO")
    logger.info(f"Created sampling client for HAPO model")
    
    # Run evaluation
    pass_at_1, average_token_length, correct_count, total_count = evaluate(sampling_client, prepared_dataset, renderer, config)
    
    logger.info(f"Final Results - Pass@1: {pass_at_1:.4f} ({correct_count}/{total_count})")
    logger.info(f"Correct count: {correct_count}")
    logger.info(f"Total count: {total_count}")
    logger.info(f"Average token length: {average_token_length:.2f} tokens")
    
    # Save results to JSON
    import datetime
    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "dataset": "ReClor",
        "data_path": config.data_path,
        "accuracy": pass_at_1,
        "correct": correct_count,
        "total": total_count,
        "avg_token_length": average_token_length
    }
    results_file = "reclor_eval_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n=== Final Results ===")
    print(f"Pass@1: {pass_at_1:.4f} ({correct_count}/{total_count})")
    print(f"Results saved to {results_file}")

if __name__ == "__main__":
    chz.nested_entrypoint(main)

