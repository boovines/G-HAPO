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
    # Default to the s1k subset test file
    data_path: str = "../datasets/test_s1k.json" 

def extract_answer_content(text: str) -> str:
    """
    Helper function to parse each type of pattern for the sample solution.
    Prioritizes explicit markers (Boxed > Phrases) over implicit content.
    """
    if not text:
        return ""
    
    text = str(text).strip()

    # Pattern 1: LaTeX Boxed Answer (Most definitive)
    # Matches \boxed{content}
    boxed_match = re.search(r'\\boxed\{(.*?)\}', text, re.DOTALL)
    if boxed_match:
        return boxed_match.group(1).strip()

    # Pattern 2: "The final answer is" phrase
    # Matches text after the phrase up to a period or end of line
    final_phrase_match = re.search(r'(?i)the final answer is[:\s]*(.*)', text)
    if final_phrase_match:
        content = final_phrase_match.group(1).strip()
        # Clean trailing punctuation if it looks like a sentence end
        if content.endswith('.'):
            content = content[:-1]
        return content

    # Pattern 3: "Answer:" or "ANSWER:" label
    answer_colon_match = re.search(r'(?i)\banswer:\s*(.*)', text)
    if answer_colon_match:
        content = answer_colon_match.group(1).strip()
        return content

    # Pattern 4: Implicit/Numerical Only
    # If no markers are found, we assume the text itself is the answer (common in s1k)
    # or we try to extract the last numerical value if it's a mix.
    # For this dataset, returning the stripped text is often the safest fallback 
    # if it's a pure numerical entry.
    return text

def normalize_answer(text: str) -> str:
    """Standardizes answer text for comparison."""
    # Remove standard LaTeX formatting that might interfere with comparison
    text = text.replace(r'\,', '').replace(',', '') # Remove thousands separators
    text = text.replace('$', '').replace('\\', '')
    return text.strip().lower()

def is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False

def is_correct_math(response: str, ground_truth_solution: str) -> bool:
    """
    Checks if the model's response matches the ground truth.
    Handles numerical equivalence and exact string matching.
    """
    # 1. Extract the specific answer content from both full texts
    model_ans_extracted = extract_answer_content(response)
    gold_ans_extracted = extract_answer_content(ground_truth_solution)
    
    # 2. Normalize
    model_norm = normalize_answer(model_ans_extracted)
    gold_norm = normalize_answer(gold_ans_extracted)

    # 3. Numerical Comparison (if both are numbers)
    if is_float(model_norm) and is_float(gold_norm):
        try:
            m_val = float(model_norm)
            g_val = float(gold_norm)
            # Use a small tolerance for floating point comparison
            return math.isclose(m_val, g_val, rel_tol=1e-5)
        except:
            pass # Fall back to string comparison

    # 4. String Comparison
    return model_norm == gold_norm

def format_s1k_question(example: Dict[str, Any]) -> str:
    """Formats the S1K question for the model."""
    # S1K questions are generally self-contained in the 'question' field
    return example["question"]

def evaluate(sampling_client, dataset, renderer, config):
    correct_count = 0
    total_count = 0
    total_token_length = 0
    
    # Batch processing could be added here, but doing sequential for clarity/template adherence
    
    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=config.eval_temperature,
        top_p=config.eval_top_p
    )

    for example in tqdm(dataset, desc="Evaluating S1K"):
        question = example["q_str"]
        ground_truth = example["ground_truth"]
        
        # Create messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]
        
        # Get response
        try:
            model_input = renderer.build_generation_prompt(messages)
            future = sampling_client.sample(
                prompt=model_input,
                num_samples=1,
                sampling_params=sampling_params
            )
            result = future.result()
            tokens = result.sequences[0].tokens
            parsed, _ = renderer.parse_response(tokens)
            response_text = parsed["content"] if parsed and "content" in parsed else ""
            
            # Calculate metrics
            total_token_length += len(tokens) # Use actual token count
            
            if is_correct_math(response_text, ground_truth):
                correct_count += 1
                
            total_count += 1
            
        except Exception as e:
            logger.error(f"Error processing example: {e}")
            continue

    accuracy = correct_count / total_count if total_count > 0 else 0
    avg_len = total_token_length / total_count if total_count > 0 else 0
    
    logger.info(f"S1K Eval Results - Accuracy: {accuracy:.2%}, Avg Length: {avg_len:.1f}")
    return accuracy, avg_len, correct_count, total_count

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
                "ground_truth": row["solution"], # The solution column contains the answer
                "metadata": row.get("cot_type", "unknown")
            })
    except ValueError:
        # Fallback for standard JSON list
        with open(config.data_path, 'r') as f:
            data = json.load(f)
            for item in data:
                prepared_dataset.append({
                    "q_str": format_s1k_question(item),
                    "ground_truth": item["solution"],
                    "metadata": item.get("cot_type", "unknown")
                })

    logger.info(f"Prepared S1K dataset: {len(prepared_dataset)} examples")
    
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
    
    # Loading specific weights as per the template
    training_client.load_state("tinker://d88742cc-b842-58bd-9c6e-16281f28b3a0:train:0/weights/000120")
    sampling_client = training_client.save_weights_and_get_sampling_client(name="HAPO")
    logger.info(f"Created sampling client for HAPO model")
    
    # Run evaluation
    pass_at_1, average_token_length, correct_count, total_count = evaluate(sampling_client, prepared_dataset, renderer, config)
    
    logger.info(f"Final Results: {correct_count}/{total_count} ({pass_at_1:.2%})")