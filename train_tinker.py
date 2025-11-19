import argparse
import os
import torch
import tinker # Assuming the library is named 'tinker'
from tinker import TensorData 
from datasets import load_dataset
from transformers import AutoTokenizer

from tinker_tracker import TinkerHistoryTracker
from utils import MAX_TRAIN_SET_SIZE

def parse_args():
    parser = argparse.ArgumentParser(description="Tinker API RL Training")
    parser.add_argument("--train_dataset", type=str, required=True)
    parser.add_argument("--tinker_api_key", type=str, required=True, help="Your Tinker API Key")
    parser.add_argument("--model_name", type=str, required=True, help="Model name on Tinker (e.g. 'Llama-3-70b')")
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4, help="Number of prompts per batch")
    parser.add_argument("--num_generations", type=int, default=4, help="Group size (G) for GRPO")
    parser.add_argument("--w_lr", type=float, default=1.0)
    # ... Add other args from your original script as needed
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Initialize Tinker Client
    client = tinker.Client(api_key=args.tinker_api_key)
    
    # Connect to the specific model/LoRA adapter
    # Note: Check Tinker docs for exact method to create/select a LoRA adapter
    training_client = client.create_lora_training_client(model=args.model_name)

    # 2. Load Data & Tokenizer (Local)
    # We need the tokenizer locally just to count tokens for rewards
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B") # Use base model equivalent
    dataset = load_dataset("json", data_files=args.train_dataset, split="train")
    
    # 3. Initialize History Tracker
    tracker = TinkerHistoryTracker(
        tokenizer=tokenizer, 
        w_lr=args.w_lr, 
        type_lr="cosine" # or args.type_lr
    )

    # 4. Training Loop
    print("Starting Tinker Training Loop...")
    
    for epoch in range(args.num_epochs):
        # Simple batching
        dataset = dataset.shuffle()
        for i in range(0, len(dataset), args.batch_size):
            batch = dataset[i : i + args.batch_size]
            prompts = batch["question"]
            ground_truths = batch["ground_truth"]
            prompt_indices = batch["id"] # Assuming your dataset has IDs, else generate them

            # --- A. SAMPLE (Remote) ---
            # We request 'num_generations' completions per prompt
            # Tinker's sample usually returns text and logprobs needed for training
            samples = client.sample(
                prompts, 
                n=args.num_generations, 
                temperature=0.7, 
                max_tokens=800,
                return_logprobs=True # Ensure we get logprobs for PPO/GRPO
            )
            
            # Flatten lists for processing
            # samples structure assumed: [[(text, logprobs), ...], ...]
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

            # --- B. COMPUTE REWARDS (Local CPU) ---
            raw_rewards = tracker.calculate_rewards(flat_pids, flat_completions, flat_gts)
            
            # --- C. COMPUTE ADVANTAGES (GRPO Logic) ---
            # Group by prompt, normalize rewards to get advantages
            advantages = []
            # Iterate in chunks of 'num_generations'
            G = args.num_generations
            for j in range(0, len(raw_rewards), G):
                group_rewards = raw_rewards[j : j + G]
                mean_r = sum(group_rewards) / G
                std_r = torch.tensor(group_rewards).std().item() + 1e-8
                
                # Standardize
                group_adv = [(r - mean_r) / std_r for r in group_rewards]
                advantages.extend(group_adv)

            # --- D. TRAIN (Remote) ---
            train_data = []
            for p, c, lp, adv in zip(flat_prompts, flat_completions, flat_logprobs, advantages):
                # Construct Tinker Datum
                # Note: "importance_sampling" loss expects target_tokens (the completion)
                # and sampling_logprobs (the logprobs when sampled)
                
                # We might need to re-tokenize the completion to get token IDs
                # or Tinker sample object might provide them.
                target_tokens = tokenizer(c, add_special_tokens=False)["input_ids"]

                datum = tinker.Datum(
                    model_input=p, # The prompt
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                        "logprobs": TensorData.from_torch(torch.tensor(lp)), # Reference logprobs
                        "advantages": TensorData.from_torch(torch.tensor(adv))
                    }
                )
                train_data.append(datum)

            # Submit to Tinker
            # "importance_sampling" implements the policy gradient step using the advantages we calculated
            client.forward_backward(train_data, loss_fn="importance_sampling") 
            
            # Apply Update
            client.optim_step()
            
            print(f"Epoch {epoch} | Step {i//args.batch_size} | Avg Reward: {sum(raw_rewards)/len(raw_rewards):.4f}")

if __name__ == "__main__":
    main()