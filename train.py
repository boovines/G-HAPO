import logging
import time
import os
from pathlib import Path
from concurrent.futures import Future

import chz
import datasets
import tinker
import torch
from tinker import types
from tinker.types.tensor_data import TensorData
import tinker_cookbook.checkpoint_utils as checkpoint_utils
import tinker_cookbook.model_info as model_info
import tinker_cookbook.renderers as renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log

# --- Custom Imports ---
from tracker import HistoryTracker
from prompting_utils import SYSTEM_PROMPT
from math_utils import is_correct
from utils import MAX_TRAIN_SET_SIZE

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)

# Suppress tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def to_list_int(data):
    """Recursively converts tensors/arrays to standard python int lists"""
    if hasattr(data, "tolist"):
        data = data.tolist()
    elif isinstance(data, torch.Tensor):
        data = data.detach().cpu().tolist()
    return [int(x) for x in data]

def to_list_float(data):
    """Recursively converts tensors/arrays to standard python float lists"""
    if hasattr(data, "tolist"):
        data = data.tolist()
    elif isinstance(data, torch.Tensor):
        data = data.detach().cpu().tolist()
    return [float(x) for x in data]

@chz.chz
class Config:
    # --- Logging ---
    wandb_project: str | None = "SEPO"
    wandb_name: str | None = "hapo_lite_100"
    
    # --- Data & Paths ---
    # LIGHTWEIGHT DATASET (100 Samples)
    train_dataset: str = "datasets/train_samples_math_100.json" 
    valid_dataset: str = "datasets/valid_samples_dsr_500.json"
    base_url: str | None = None
    log_path: str = "/tmp/tinker-experiments/grpo-math-4b"
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"

    batch_size: int = 4
    group_size: int = 4
    learning_rate: float = 1e-5
    lora_rank: int = 8
    save_every: int = 10
    max_tokens: int = 800

    # Reward Hyperparams
    w_lr: float = 1.0
    type_lr: str = "cosine"
    temperature: float = 0.7
    mode: str = "min" # 'min' or 'mean'
    rep_ngram_size: int = 3
    rep_penalty: float = 0.0
    
    # Training Hyperparams
    gradient_accumulation_steps: int = 1
    num_epochs: int = 1
    beta: float = 0.1  # KL penalty coefficient
    
    # API
    tinker_api_key: str | None = None


def load_env_file():
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def evaluate(client, dataset, renderer, config):
    logger.info("Running validation evaluation...")
    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=config.temperature
    )
    
    correct_count = 0
    total_count = 0
    
    for example in dataset:
        q_str = example["q_str"]
        ground_truth = example["ground_truth"]
        
        convo = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q_str}
        ]
        model_input = renderer.build_generation_prompt(convo)
        
        # Sample 4 times for pass@1
        sample_futures = []
        for _ in range(4):
            sample_futures.append(
                client.sample(
                    prompt=model_input,
                    num_samples=1,
                    sampling_params=sampling_params,
                )
            )
        
        # Check if any of the 4 samples is correct
        is_correct_any = False
        for future in sample_futures:
            res = future.result()
            s_tokens = res.sequences[0].tokens
            parsed_msg, _ = renderer.parse_response(s_tokens)
            content = parsed_msg["content"] if parsed_msg and "content" in parsed_msg else ""
            
            if is_correct(content, ground_truth):
                is_correct_any = True
                break
        
        if is_correct_any:
            correct_count += 1
        total_count += 1
    
    pass_at_1 = correct_count / total_count if total_count > 0 else 0.0
    logger.info(f"Validation Pass@1: {pass_at_1:.4f} ({correct_count}/{total_count})")
    return pass_at_1


def main(config: Config):
    # Load .env file if it exists
    load_env_file()

    # Set up API key
    if config.tinker_api_key:
        os.environ["TINKER_API_KEY"] = config.tinker_api_key
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name,
        config=config,
        do_configure_logging_module=True,
    )

    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")

    tracker = HistoryTracker(
        tokenizer=tokenizer, 
        w_lr=config.w_lr, 
        type_lr=config.type_lr,
        rep_ngram_size=config.rep_ngram_size,
        rep_penalty=config.rep_penalty,
        mode=config.mode
    )

    # --- Data Loading ---
    def prepare_example(example, idx):
        q_str = " ".join(str(item) for item in example["question"]) if isinstance(example["question"], list) else example["question"]
        return {
            "q_str": q_str,
            "ground_truth": example["ground_truth"],
            "prompt_idx": idx
        }

    logger.info(f"Loading training data: {config.train_dataset}")
    train_dataset = datasets.load_dataset("json", data_files=config.train_dataset, split="train")
    train_dataset = train_dataset.map(prepare_example, with_indices=True)

    logger.info(f"Loading validation data: {config.valid_dataset}")
    valid_dataset = datasets.load_dataset("json", data_files=config.valid_dataset, split="train")
    valid_dataset = valid_dataset.map(prepare_example, with_indices=True)

    # --- Client Setup ---
    service_client = tinker.ServiceClient(base_url=config.base_url)

    logger.info("Initializing Student Model Client...")
    training_client = service_client.create_lora_training_client(
        base_model=config.model_name,
        rank=config.lora_rank
    )

    logger.info("Initializing Reference Model Client from Init Weights...")
    init_weights_path = training_client.save_weights_for_sampler(name="init").result().path
    ref_client = service_client.create_sampling_client(model_path=init_weights_path)

    best_pass_at_1 = 0.0
    start_global_step = 0

    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=config.temperature
    )
    
    # Note: adam_params will be instantiated per step to handle LR scheduling

    steps_per_epoch = (len(train_dataset) // config.batch_size) // config.gradient_accumulation_steps
    total_optimizer_steps = steps_per_epoch * config.num_epochs
    
    logger.info(f"Training for {config.num_epochs} epochs.")
    logger.info(f"Steps per epoch: {steps_per_epoch}. Total optimizer steps: {total_optimizer_steps}")
    
    global_step = start_global_step
    accum_metrics = {"reward": [], "kl": []}
    
    for epoch in range(config.num_epochs):
        logger.info(f"Starting Epoch {epoch + 1}/{config.num_epochs}")
        shuffled_dataset = train_dataset.shuffle(seed=42 + epoch)
        n_batches_per_epoch = len(shuffled_dataset) // config.batch_size
        
        for batch_i in range(n_batches_per_epoch):
            current_optimizer_step = (batch_i + (epoch * n_batches_per_epoch)) // config.gradient_accumulation_steps
            if current_optimizer_step < start_global_step:
                continue

            is_start_of_accum = (batch_i % config.gradient_accumulation_steps == 0)
            
            # --- Checkpointing & Eval ---
            if is_start_of_accum and global_step % config.save_every == 0 and global_step > 0:
                logger.info(f"Checkpointing at step {global_step}...")
                
                checkpoint_utils.save_checkpoint(
                    training_client=training_client,
                    name=f"{global_step:06d}",
                    log_path=config.log_path,
                    kind="state",
                    loop_state={"batch": global_step},
                )
                tracker.save_state(os.path.join(config.log_path, f"tracker_{global_step:06d}.pkl"))

                # SHORTCUT METHOD
                eval_client = training_client.save_weights_and_get_sampling_client(name="eval_temp")
                curr_pass_at_1 = evaluate(eval_client, valid_dataset, renderer, config)
                ml_logger.log_metrics({"eval/pass_at_1": curr_pass_at_1}, step=global_step)

                if curr_pass_at_1 > best_pass_at_1:
                    logger.info(f"New best accuracy: {curr_pass_at_1:.4f}. Saving best_checkpoint.")
                    best_pass_at_1 = curr_pass_at_1
                    checkpoint_utils.save_checkpoint(
                        training_client=training_client,
                        name="best_checkpoint",
                        log_path=config.log_path,
                        kind="both",
                        loop_state={"batch": global_step, "accuracy": best_pass_at_1},
                    )

            # --- Load Micro-Batch ---
            batch_start = batch_i * config.batch_size
            batch_end = min((batch_i + 1) * config.batch_size, len(shuffled_dataset))
            batch_rows = shuffled_dataset.select(range(batch_start, batch_end))

            # SHORTCUT METHOD
            sampling_client = training_client.save_weights_and_get_sampling_client(name="latest")

            batch_futures = []
            batch_prompts_tokens = []

            for q_str in batch_rows["q_str"]:
                convo = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": q_str}
                ]
                model_input = renderer.build_generation_prompt(convo)
                prompt_tokens = model_input.to_ints()

                sample_futures = []
                for _ in range(config.group_size):
                    sample_futures.append(
                        sampling_client.sample(
                            prompt=model_input,
                            num_samples=1,
                            sampling_params=sampling_params,
                        )
                    )
                batch_futures.append(sample_futures)
                batch_prompts_tokens.append(prompt_tokens)

            # Process Batch
            batch_datums = []
            batch_rewards_log = []
            batch_kl_log = []

            for sample_futures, prompt_tokens, ground_truth, pid in zip(
                batch_futures, batch_prompts_tokens, batch_rows["ground_truth"], batch_rows["prompt_idx"]
            ):
                group_tokens = []
                group_logprobs = []
                group_ob_lens = []
                group_texts = []
                
                for future in sample_futures:
                    res = future.result()
                    s_tokens = res.sequences[0].tokens
                    s_logprobs = res.sequences[0].logprobs 
                    
                    full_tokens = prompt_tokens + s_tokens
                    group_tokens.append(full_tokens)
                    group_logprobs.append(s_logprobs)
                    group_ob_lens.append(len(prompt_tokens) - 1)
                    
                    parsed_msg, _ = renderer.parse_response(s_tokens)
                    content = parsed_msg["content"] if parsed_msg and "content" in parsed_msg else ""
                    group_texts.append(content)

                # --- REFERENCE FORWARD PASS (Use compute_logprobs list directly) ---
                ref_logprobs_sums = []
                for seq_tokens, ob_len in zip(group_tokens, group_ob_lens):
                    full_tokens_safe = to_list_int(seq_tokens)
                    
                    try:
                        ref_model_input = types.ModelInput.from_ints(tokens=full_tokens_safe)
                        logprobs_list = ref_client.compute_logprobs(ref_model_input).result()
                        
                        if not logprobs_list:
                            # Fallback to avoid crash if ref model fails
                            ref_sum = sum(group_logprobs[group_tokens.index(seq_tokens)])
                        else:
                            prompt_len = len(prompt_tokens)
                            if len(logprobs_list) > prompt_len:
                                response_logprobs = logprobs_list[prompt_len:]
                                clean_response_logprobs = [lp if lp is not None else 0.0 for lp in response_logprobs]
                                ref_sum = sum(clean_response_logprobs)
                            else:
                                ref_sum = 0.0
                        
                        ref_logprobs_sums.append(ref_sum)
                        
                    except Exception as e:
                        logger.error(f"DEBUG: ref_client.compute_logprobs FAILED. Error: {e}")
                        raise e

                # Rewards
                pids_expanded = [pid] * len(group_texts)
                gts_expanded = [ground_truth] * len(group_texts)
                
                base_rewards = tracker.calculate_rewards(
                    pids_expanded, group_texts, gts_expanded
                )
                
                final_rewards = []
                for i in range(len(base_rewards)):
                    sampled_sum = sum(group_logprobs[i])
                    ref_sum = ref_logprobs_sums[i]
                    
                    # KL = Student - Teacher
                    kl_val = sampled_sum - ref_sum
                    
                    penalized_reward = base_rewards[i] - (config.beta * kl_val)
                    final_rewards.append(penalized_reward)
                    batch_kl_log.append(kl_val)

                batch_rewards_log.extend(final_rewards)

                rewards_t = torch.tensor(final_rewards, dtype=torch.float32)
                mean = rewards_t.mean()
                std = rewards_t.std(unbiased=False) + 1e-8
                advantages = ((rewards_t - mean) / std).tolist()

                for tokens, logprob, adv, ob_len in zip(
                    group_tokens, group_logprobs, advantages, group_ob_lens
                ):
                    input_tokens = tokens[:-1]
                    target_tokens = tokens[1:]
                    
                    target_tokens_safe = to_list_int(target_tokens)
                    all_logprobs_safe = to_list_float([0.0] * ob_len + logprob)
                    all_advantages_safe = to_list_float([0.0] * ob_len + [adv] * (len(input_tokens) - ob_len))
                    
                    datum = types.Datum(
                        model_input=types.ModelInput.from_ints(tokens=input_tokens),
                        loss_fn_inputs={
                            "target_tokens": target_tokens_safe,
                            "logprobs": all_logprobs_safe,
                            "advantages": all_advantages_safe,
                        },
                    )
                    batch_datums.append(datum)
                
                tracker.update_batch_history(pids_expanded, group_texts, gts_expanded)

            # --- Accumulate Gradients Immediately ---
            if batch_datums:
                try:
                    _ = training_client.forward_backward(
                        batch_datums, "importance_sampling"
                    ).result()
                except Exception as e:
                    logger.error(f"DEBUG: training_client.forward_backward FAILED. Error: {e}")
                    raise e
            
            avg_reward = sum(batch_rewards_log) / len(batch_rewards_log) if batch_rewards_log else 0.0
            avg_kl = sum(batch_kl_log) / len(batch_kl_log) if batch_kl_log else 0.0
            accum_metrics["reward"].append(avg_reward)
            accum_metrics["kl"].append(avg_kl)

            if ((batch_i + 1) % config.gradient_accumulation_steps == 0) or ((batch_i + 1) == n_batches_per_epoch):
                
                # --- LINEAR LR SCHEDULER ---
                progress = global_step / total_optimizer_steps
                current_lr = config.learning_rate * (1.0 - progress)
                current_lr = max(0.0, current_lr)
                
                # Create NEW immutable AdamParams object for each step
                current_adam_params = types.AdamParams(
                    learning_rate=current_lr, 
                    beta1=0.9, 
                    beta2=0.95, 
                    eps=1e-8
                )

                _ = training_client.optim_step(current_adam_params).result()
                
                final_reward = sum(accum_metrics["reward"]) / len(accum_metrics["reward"]) if accum_metrics["reward"] else 0.0
                final_kl = sum(accum_metrics["kl"]) / len(accum_metrics["kl"]) if accum_metrics["kl"] else 0.0
                
                avg_kl_per_token = final_kl / 200.0 # Approx
                
                metrics = {
                    "progress/global_step": global_step,
                    "progress/epoch": epoch + 1,
                    "optim/lr": current_lr,
                    "reward/mean": final_reward,
                    "metrics/kl": final_kl
                }
                ml_logger.log_metrics(metrics, step=global_step)
                logger.info(f"Step {global_step} | LR: {current_lr:.2e} | Reward: {final_reward:.4f} | KL (Seq): {final_kl:.4f} | ~KL/tok: {avg_kl_per_token:.4f}")

                accum_metrics = {"reward": [], "kl": []}
                global_step += 1

    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name="final",
        log_path=config.log_path,
        kind="both",
        loop_state={"batch": global_step},
    )
    tracker.save_state(os.path.join(config.log_path, "tracker_final.pkl"))
    ml_logger.close()
    logger.info("Training completed")

if __name__ == "__main__":
    chz.nested_entrypoint(main)