import chz
import json
from datasets import load_dataset
import tinker
from tinker import types
import tinker_cookbook.model_info as model_info
import tinker_cookbook.renderers as renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
import logging
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
from math_utils import is_correct

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)

@chz.chz
class Config:
    max_tokens: int = 8192
    eval_temperature: float = 0.6
    eval_top_p: float = 0.95
    eval_batch_size: int = 4


def evaluate(client, dataset, renderer, config):
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
        batch = dataset.select(range(i * config.eval_batch_size, min((i + 1) * config.eval_batch_size, len(dataset))))
        
        batch_size = len(batch)
        prompt_futures = []
        ground_truths = batch["ground_truth"]
        
        for q_str in batch["q_str"]:
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
                
                if is_correct(content, gt, use_math_verify=True):
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


def main(config: Config):
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load MATH-500 dataset
    math_500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
    
    # Randomly sample 50 examples
    math_500 = math_500.shuffle(seed=42).select(range(50))
    logger.info(f"Randomly sampled 50 examples from MATH-500 dataset")
    
    # Prepare dataset format
    def prepare_example(example):
        return {
            "q_str": example["problem"],
            "ground_truth": example["answer"]
        }
    
    math_500 = math_500.map(prepare_example)
    logger.info(f"Prepared MATH-500 dataset: {len(math_500)} examples")
    
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
    pass_at_1, average_token_length, correct_count, total_count = evaluate(sampling_client, math_500, renderer, config)
    
    logger.info(f"Final Results - Pass@1: {pass_at_1:.4f} ({correct_count}/{total_count})")
    logger.info(f"Correct count: {correct_count}")
    logger.info(f"Total count: {total_count}")
    logger.info(f"Average token length: {average_token_length:.2f} tokens")
    
    # Save results to JSON
    import datetime
    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "dataset": "MATH-500",
        "accuracy": pass_at_1,
        "correct": correct_count,
        "total": total_count,
        "avg_token_length": average_token_length
    }
    results_file = "math500_eval_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n=== Final Results ===")
    print(f"Pass@1: {pass_at_1:.4f} ({correct_count}/{total_count})")
    print(f"Results saved to {results_file}")

if __name__ == "__main__":
    chz.nested_entrypoint(main)

