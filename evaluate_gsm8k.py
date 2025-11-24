import chz
from datasets import load_dataset
import tinker
import torch
from tinker import types
from tinker.types.tensor_data import TensorData
import tinker_cookbook.checkpoint_utils as checkpoint_utils
import tinker_cookbook.model_info as model_info
import tinker_cookbook.renderers as renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
import logging
from tqdm import tqdm

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
        batch = dataset.select(range(
            i * config.eval_batch_size, 
            min((i + 1) * config.eval_batch_size, len(dataset))
        ))
        
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
        for future, gt in zip(prompt_futures, ground_truths):
            try:
                result = future.result()
                tokens = result.sequences[0].tokens
                parsed, _ = renderer.parse_response(tokens)
                content = parsed["content"] if parsed and "content" in parsed else ""
                
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
            
    pass_at_1 = correct_count / total_count
    average_token_length = sum(token_lengths) / len(token_lengths)
    return pass_at_1, average_token_length, correct_count, total_count


def main(config: Config):
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load GSM8K test dataset
    gsm8k_test = load_dataset("openai/gsm8k", "main", split="test")
    
    # Randomly sample 50 examples
    gsm8k_test = gsm8k_test.shuffle(seed=42).select(range(50))
    logger.info(f"Randomly sampled 50 examples from GSM8K test set")
    
    # Prepare dataset format
    def prepare_example(example):
        return {
            "q_str": example["question"],
            "ground_truth": example["answer"].split("####")[-1].strip()
        }
    
    gsm8k_test = gsm8k_test.map(prepare_example)
    logger.info(f"Loaded GSM8K test dataset: {len(gsm8k_test)} examples")
    
    # Model setup
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    
    # Initialize tokenizer and renderer
    tokenizer = get_tokenizer(model_name)
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")
    
    # Create sampling client
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=model_name)
    logger.info(f"Created sampling client for model: {model_name}")
    
    # Run evaluation
    pass_at_1, average_token_length, correct_count, total_count = evaluate(sampling_client, gsm8k_test, renderer, config)
    
    logger.info(f"Final Results - Pass@1: {pass_at_1:.4f} ({correct_count}/{total_count})")
    logger.info(f"Correct count: {correct_count}")
    logger.info(f"Total count: {total_count}")
    logger.info(f"Average token length: {average_token_length:.2f} tokens")
    

if __name__ == "__main__":
    chz.nested_entrypoint(main)