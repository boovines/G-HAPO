import math
import numpy as np

from math_utils import is_correct 
from utils import MAX_LEN, compute_len_reward, compute_len_reward_linear, compute_repetition_penalty_reward

class HistoryTracker:
    def __init__(self, tokenizer, w_lr=1.0, type_lr="cosine", rep_ngram_size=3, rep_penalty=0.0, mode="min"):
        self.tokenizer = tokenizer
        self.w_lr = w_lr
        self.type_lr = type_lr
        self.rep_ngram_size = rep_ngram_size
        self.rep_penalty = rep_penalty
        self.mode = mode
        
        # Initialize state
        self.lens_dict = {} 
        self.count_dict = {}
        
        # Config
        self.consider_readability = False
        self.tolerance_ratio = 0.1
        self.use_math_verify = True

        # Select Reward Function
        if type_lr == "cosine":
            self.compute_lr_func = compute_len_reward
        elif type_lr == "linear":
            self.compute_lr_func = compute_len_reward_linear
        else:
            raise ValueError(f"Unknown type_lr: {type_lr}")

    def update_history(self, prompt_idx, correct_lengths):
        """Updates the history of best lengths based on new correct completions."""
        if not correct_lengths:
            return

        current_val = self.lens_dict.get(prompt_idx, MAX_LEN)
        
        if self.mode == "min":
            new_min = min(correct_lengths)
            self.lens_dict[prompt_idx] = min(current_val, new_min)
            
        elif self.mode == "mean":
            current_count = self.count_dict.get(prompt_idx, 0)
            batch_sum = sum(correct_lengths)
            batch_count = len(correct_lengths)
            
            if current_count == 0:
                self.lens_dict[prompt_idx] = batch_sum / batch_count
            else:
                prev_total = current_val * current_count
                new_total = prev_total + batch_sum
                self.lens_dict[prompt_idx] = new_total / (current_count + batch_count)
            
            self.count_dict[prompt_idx] = current_count + batch_count

    def calculate_rewards(self, prompt_indices, completions, ground_truths):
        """
        Calculates the combined reward (Correctness + Length + Repetition)
        and updates the history tracker.
        """
        rewards = []
        # Tokenize all completions at once
        encodings = self.tokenizer(completions, add_special_tokens=False)
        completion_tokens = encodings["input_ids"]

        for i, (resp, tokens, gt, pid) in enumerate(zip(completions, completion_tokens, ground_truths, prompt_indices)):
            # 1. Correctness
            is_corr = is_correct(resp, gt, use_math_verify=self.use_math_verify)
            r_corr = float(is_corr)

            # 2. Length Reward
            history_len = self.lens_dict.get(pid, MAX_LEN)
            seq_len = len(tokens)
            
            r_len = self.compute_lr_func(
                history_len=history_len,
                seq_len=seq_len,
                is_corr=r_corr >= 1.0,
                prompt_idx=pid,
                consider_readability=self.consider_readability,
                tolerance_ratio=self.tolerance_ratio,
                w_lr=self.w_lr
            )

            # 3. Repetition Reward
            r_rep = compute_repetition_penalty_reward(tokens, self.rep_ngram_size, self.rep_penalty)

            total_reward = r_corr + r_len + r_rep
            rewards.append(total_reward)

            # Update History (if correct)
            if r_corr >= 1.0:
                self.update_history(pid, [seq_len])

        return rewards