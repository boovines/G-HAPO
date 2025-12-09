import math

MAX_LEN = 100000

def compute_len_reward_linear(history_len, seq_len, is_corr, prompt_idx, consider_readability: bool, tolerance_ratio: float, w_lr: float):
    """
    Calculates a linear reward based on the ratio between the current sequence length
    and the historical best length.
    """
    if history_len < MAX_LEN and history_len > 0:
        # h_i is not Null (we have seen a correct answer before)
        if is_corr:
            if seq_len < history_len:
                # Reward for being shorter than history
                ratio_len = seq_len / history_len
                if consider_readability:
                    raise NotImplementedError
                else:
                    r = 1.0 - ratio_len
            elif seq_len >= history_len and seq_len < (1+tolerance_ratio) * history_len:
                # Neutral zone (within tolerance)
                r = 0
            else:
                # Penalty for being longer than history
                unconstrained_r = -1.0 * ((seq_len - (1+tolerance_ratio) * history_len) / history_len)
                # Cap penalty for correct answers so they aren't worse than incorrect ones
                r = max(-0.7, unconstrained_r)
        else:
            # Incorrect answer logic
            if seq_len < (1+tolerance_ratio) * history_len:
                r = 0
            else:
                unconstrained_r = -1.0 * ((seq_len - (1+tolerance_ratio) * history_len) / history_len)
                r = max(-1.0, unconstrained_r)
    else:
        # h_i is Null (first time seeing this prompt or no correct answers yet)
        r = 0
        
    return r * w_lr

def compute_len_reward(history_len, seq_len, is_corr, prompt_idx, consider_readability: bool, tolerance_ratio: float, w_lr: float):
    """
    Calculates length reward using a cosine schedule for a smoother gradient.
    """
    if history_len < MAX_LEN and history_len > 0:
        # h_i is not Null
        if seq_len < 2 * history_len:
            r = math.cos((seq_len/history_len) * (math.pi/2))
        else:
            r = math.cos(math.pi) # Which is -1

        if is_corr:
            r = max(-0.7, r)
        else:
            r = min(0, r)
    else:
        # h_i is Null
        r = 0
        
    return r * w_lr

# Source:
# https://stackoverflow.com/questions/21883108/fast-optimize-n-gram-implementations-in-python
def zipngram(tokens: list[int], ngram_size: int):
    return zip(*[tokens[i:] for i in range(ngram_size)])

def compute_repetition_penalty_reward(resp_tokens: list[int], ngram_size: int, max_penalty: float, only_start: bool = False) -> float:
    """
    Reward function that penalizes repetitions.
    Args:
        resp_tokens: Token ids of a single response
        ngram_size: The size of the n-grams to check for repetition (e.g., 3)
        max_penalty: The negative value to apply per repeated token (e.g., -0.1)
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    if max_penalty == 0.0 or len(resp_tokens) < ngram_size:
        return 0.0
    
    # Find repeated n-grams and their positions
    repeated_positions = []
    ngrams = set()

    for start_idx, ng in enumerate(zipngram(resp_tokens, ngram_size)):
        if ng in ngrams:
            repeated_positions.append(start_idx)
        ngrams.add(ng)

    # Calculate word-level penalties
    word_penalties = [0.0] * len(resp_tokens)
    curr_end_idx = -1

    for start_idx in repeated_positions:
        if not only_start or start_idx > curr_end_idx:
            # Apply penalty to each token in the repeated n-gram
            for i in range(start_idx, start_idx + ngram_size):
                word_penalties[i] = max_penalty
        curr_end_idx = start_idx + ngram_size - 1

    # Average the word-level penalties for the final reward
    reward = sum(word_penalties) / len(word_penalties) if word_penalties else 0.0

    return reward

def compute_repetition_penalty_reward_1(resp_tokens: list[int], ngram_size: int, max_penalty: float) -> float:
    """
    Alternative repetition penalty calculation.
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    if max_penalty == 0.0:
        return 0.0

    ngrams = set()
    total = 0
    for ng in zipngram(resp_tokens, ngram_size):
        ngrams.add(ng)
        total += 1

    scaling = 1 - len(ngrams) / total
    return scaling * max_penalty