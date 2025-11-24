import argparse
import os
import torch
import tinker
from tinker import TensorData 
from datasets import load_dataset
from transformers import AutoTokenizer

# Custom imports from project files
from tinker_tracker import TinkerHistoryTracker
from prompting_utils import SYSTEM_PROMPT
from utils import MAX_TRAIN_SET_SIZE

# Suppress tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    parser = argparse.ArgumentParser(description="Tinker API RL Training")
    
    # Dataset and API Config
    parser.add_argument("--train_dataset", type=str, required=True, help="Path to training dataset JSON")
    
    # Model Configuration
    parser.add_argument("--model_name", type=str, required=True, 
                        help="The model ID on Tinker (e.g. 'meta-llama/Meta-Llama-3-70B')")
    parser.add_argument("--tokenizer_name", type=str, default=None,
                        help="HF Hub path for the tokenizer. If not set, uses --model_name.")

    # Training Hyperparameters
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4, help="Number of prompts per batch")
    parser.add_argument("--num_generations", type=int, default=4, help="Group size (G) for GRPO")
    parser.add_argument("--w_lr", type=float, default=1.0, help="Weight for the length reward")
    parser.add_argument("--temperature", type=float, default=0.7)
    
    return parser.parse_args()

def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Load Local Tokenizer
    # ------------------------------------------------------------------
    # We need this locally to count tokens for the length reward.
    # We allow a separate tokenizer_name in case the API model uses a custom alias.
    hub_path = args.tokenizer_name if args.tokenizer_name else args.model_name
    print(f"Loading local tokenizer from: {hub_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(hub_path)
        # Ensure pad_token exists for batch processing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except OSError:
        raise ValueError(f"Could not load tokenizer from '{hub_path}'. Please check your --tokenizer_name argument.")

    # ------------------------------------------------------------------
    # 2. Load and Format Data (Applying Chat Templates)
    # ------------------------------------------------------------------
    print(f"Loading dataset from {args.train_dataset}...")
    dataset = load_dataset("json", data_files=args.train_dataset, split="train")

    def add_prefix(example, idx):
        # Handle edge case where question might be a list of strings
        if isinstance(example["question"], list):
            question_str = " ".join(str(item) for item in example["question"])
        else:
            question_str = example["question"]

        # Apply the exact chat template using the imported SYSTEM_PROMPT
        prompt = tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": question_str
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        
        return {
            "prompt": prompt,             # The fully formatted conversation string
            "ground_truth": example["ground_truth"],
            "prompt_idx": idx,            # Keep ID for the history tracker
        }

    print("Formatting dataset with chat templates...")
    # map() processes the data and 'remove_columns' ensures we don't accidentally use raw data later
    dataset = dataset.map(add_prefix, with_indices=True, remove_columns=dataset.column_names)

    # ------------------------------------------------------------------
    # 3. Initialize Tinker Client & History Tracker
    # ------------------------------------------------------------------
    service_client = tinker.ServiceClient()
    print(f"Initializing LoRA training client for {args.model_name}...")
    training_client = service_client.create_lora_training_client(base_model=args.model_name)
    
    # Initialize the custom reward tracker (logic from grpo_trainer.py)
    tracker = TinkerHistoryTracker(
        tokenizer=tokenizer, 
        w_lr=args.w_lr, 
        type_lr="cosine"
    )

    # ------------------------------------------------------------------
    # 4. Main Training Loop
    # ------------------------------------------------------------------
    print("Starting Tinker Training Loop...")
    
    for epoch in range(args.num_epochs):
        dataset = dataset.shuffle()
        
        # Iterate through the dataset in batches
        for i in range(0, len(dataset), args.batch_size):
            batch = dataset[i : i + args.batch_size]
            
            prompts = batch["prompt"] 
            ground_truths = batch["ground_truth"]
            prompt_indices = batch["prompt_idx"]

            # --- A. SAMPLE (Remote on Tinker) ---
            # We send the formatted prompts. The API generates 'n' completions per prompt.
            samples = training_client.sample(
                prompts, 
                n=args.num_generations, 
                temperature=args.temperature, 
                max_tokens=800,
                return_logprobs=True 
            )
            
            # Flatten the batch structure for processing
            flat_prompts = []
            flat_completions = []
            flat_gts = []
            flat_pids = []
            flat_logprobs = []
            
            for p, g, pid, prompt_samples in zip(prompts, ground_truths, prompt_indices, samples):
                for s in prompt_samples:
                    flat_prompts.append(p)
                    flat_completions.append(s.text)
                    flat_logprobs.append(s.logprobs)
                    flat_gts.append(g)
                    flat_pids.append(pid)

            # --- B. COMPUTE REWARDS (Local on CPU) ---
            # Calculates: Correctness (1.0/0.0) + Length Reward (Dynamic) + Repetition Penalty
            raw_rewards = tracker.calculate_rewards(flat_pids, flat_completions, flat_gts)
            
            # --- C. COMPUTE GRPO ADVANTAGES (Local Math) ---
            # Normalize rewards relative to the group of 'num_generations'
            advantages = []
            G = args.num_generations
            for j in range(0, len(raw_rewards), G):
                group_rewards = raw_rewards[j : j + G]
                
                if len(group_rewards) > 1:
                    mean_r = sum(group_rewards) / len(group_rewards)
                    std_r = torch.tensor(group_rewards).std().item() + 1e-8
                    group_adv = [(r - mean_r) / std_r for r in group_rewards]
                else:
                    # Fallback if G=1 (standard RL, not GRPO)
                    group_adv = [0.0] 
                
                advantages.extend(group_adv)

            # --- D. TRAIN (Remote on Tinker) ---
            train_data = []
            for p, c, lp, adv in zip(flat_prompts, flat_completions, flat_logprobs, advantages):
                # Tokenize the target completion locally to create the datum
                target_tokens = tokenizer(c, add_special_tokens=False)["input_ids"]

                datum = tinker.Datum(
                    model_input=p,
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                        "logprobs": TensorData.from_torch(torch.tensor(lp)),
                        "advantages": TensorData.from_torch(torch.tensor(adv))
                    }
                )
                train_data.append(datum)

            # Submit gradients to the remote model
            training_client.forward_backward(train_data, loss_fn="importance_sampling") 
            training_client.optim_step()
            
            # Logging
            avg_reward = sum(raw_rewards)/len(raw_rewards)
            print(f"Epoch {epoch+1} | Batch {i//args.batch_size} | Avg Reward: {avg_reward:.4f}")

if __name__ == "__main__":
    main()