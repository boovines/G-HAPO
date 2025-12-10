import logging
import os

import chz
from datasets import load_dataset
import tinker
import torch
from tinker import types
import tinker_cookbook.checkpoint_utils as checkpoint_utils
import tinker_cookbook.model_info as model_info
import tinker_cookbook.renderers as renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log

from merged_tracker import HistoryTracker
from prompting_utils import SYSTEM_PROMPT
from merged_math_utils import is_correct 

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@chz.chz
class Config:
    # --- Logging ---
    wandb_project: str | None = "SEPO"
    wandb_name: str | None = "hapo_merged"
    
    # --- Data & Paths ---
    train_dataset: str = "merged_train.json" 
    valid_dataset: str = "merged_val.json"
    log_path: str = "../tinker-experiments/hapo-merged"

    # --- Model ---
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    
    # --- Training Hyperparameters ---
    num_epochs: int = 10 
    learning_rate: float = 1e-5 
    batch_size: int = 2           
    group_size: int = 2           
    gradient_accumulation_steps: int = 4 
    max_tokens: int = 8192       
    lora_rank: int = 8

    # Save every 5 epochs (60 steps)
    save_every: int = 60          
    
    # --- Reward & Algorithm Parameters ---
    beta: float = 0.04            
    w_lr: float = 1.0             
    clip_c: float = -0.7          
    type_lr: str = "cosine"       
    mode: str = "min"             
    
    # --- Sampling ---
    temperature: float = 0.7   

    # --- Evaluation Params ---   
    eval_temperature: float = 0.6 
    eval_top_p: float = 0.95      
    eval_batch_size: int = 4

    # --- Repetition Penalty ---
    rep_ngram_size: int = 3
    rep_penalty: float = 0.0

# --- Unified Data Processing ---
def prepare_merged_example(example, idx):
    """
    detects source and applies Tagging for the router.
    """
    source = example.get("dataset_source", "HAPO") # Default to HAPO if missing
    q_str = ""
    tagged_gt = ""

    if source == "ReClor":
        # Format: Context + Question + Options
        options = example.get('answers', [])
        labels = ['A', 'B', 'C', 'D']
        opt_str = "\n".join([f"{l}) {opt}" for l, opt in zip(labels, options)])
        q_str = f"{example.get('context','')}\n\n{example.get('question','')}\n\nOptions:\n{opt_str}\n\nAnswer:"
        
        # Convert numeric label to Letter
        label = example.get("label")
        real_gt = chr(ord('A') + int(label)) if label is not None else ""
        tagged_gt = f"RECLOR:::{real_gt}"

    elif source == "s1K":
        q_str = example.get("question", "")
        # s1K answers are in 'solution' usually
        real_gt = str(example.get("solution", ""))
        tagged_gt = f"S1K:::{real_gt}"

    else: # HAPO
        q_str = example.get("question", "")
        real_gt = str(example.get("ground_truth", ""))
        tagged_gt = f"HAPO:::{real_gt}"

    return {
        "q_str": q_str,
        "ground_truth": tagged_gt,
        "prompt_idx": idx
    }

def evaluate(client, dataset, renderer, config):
    logger.info("Running validation evaluation...")
    sampling_params = types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=config.eval_temperature,
        top_p=config.eval_top_p
    )

    correct_count = 0
    total_count = 0

    n_batches = (len(dataset) + config.eval_batch_size - 1) // config.eval_batch_size
    
    for i in range(n_batches):
        batch = dataset.select(range(i * config.eval_batch_size, min((i + 1) * config.eval_batch_size, len(dataset))))

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
            
        for future, gt in zip(prompt_futures, ground_truths):
            try:
                result = future.result()
                tokens = result.sequences[0].tokens
                parsed, _ = renderer.parse_response(tokens)
                content = parsed["content"]
                
                if is_correct(content, gt, use_math_verify=True):
                    correct_count += 1
            except Exception as e:
                logger.error(f"Error during eval sample: {e}")

            total_count += 1
    
    try:
        pass_at_1 = correct_count / total_count
    except ZeroDivisionError:
        logger.error(f"Error calculating pass@1: total_count = 0")

    logger.info(f"Validation Pass@1: {pass_at_1:.4f} ({correct_count}/{total_count})")

    return pass_at_1

def main(config: Config):
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
    logger.info(f"Loading merged train data: {config.train_dataset}")
    train_dataset = load_dataset("json", data_files=config.train_dataset, split="train")
    train_dataset = train_dataset.map(prepare_merged_example, with_indices=True)

    logger.info(f"Loading merged valid data: {config.valid_dataset}")
    valid_dataset = load_dataset("json", data_files=config.valid_dataset, split="train")
    valid_dataset = valid_dataset.map(prepare_merged_example, with_indices=True)

    # --- Client Setup ---
    service_client = tinker.ServiceClient()

    logger.info("Initializing Student Model Client...")
    training_client = service_client.create_lora_training_client(base_model=config.model_name, rank=config.lora_rank)

    logger.info("Initializing Reference Model Client from Init Weights...")
    init_weights_path = training_client.save_weights_for_sampler(name="init").result().path
    ref_client = service_client.create_sampling_client(model_path=init_weights_path)

    best_pass_at_1 = 0.0
    start_global_step = 0

    sampling_params = types.SamplingParams(
        max_tokens=config.max_tokens, 
        stop=renderer.get_stop_sequences(), 
        temperature=config.temperature
    )
    
    steps_per_epoch = (len(train_dataset) // config.batch_size) // config.gradient_accumulation_steps
    total_optimizer_steps = steps_per_epoch * config.num_epochs

    global_step = start_global_step
    accum_metrics = {"reward": [], "kl": []}
    
    logger.info(f"Starting Training: {config.num_epochs} Epochs")

    for epoch in range(config.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{config.num_epochs}")
        shuffled_dataset = train_dataset.shuffle(seed=42 + epoch)
        n_batches_per_epoch = len(shuffled_dataset) // config.batch_size
        
        for batch_i in range(n_batches_per_epoch):
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
                
                eval_client = training_client.save_weights_and_get_sampling_client(name="eval_temp")
                curr_pass_at_1 = evaluate(eval_client, valid_dataset, renderer, config)
                ml_logger.log_metrics({"eval/pass_at_1": curr_pass_at_1}, step=global_step)
                
                if curr_pass_at_1 > best_pass_at_1:
                    logger.info(f"New best accuracy: {curr_pass_at_1:.4f}")
                    best_pass_at_1 = curr_pass_at_1
                    checkpoint_utils.save_checkpoint(
                        training_client=training_client, 
                        name=f"best_checkpoint_{global_step}",
                        log_path=config.log_path, 
                        kind="both", 
                        loop_state={"batch": global_step, "acc": best_pass_at_1}
                    )

            # --- Load Batch ---
            batch_start = batch_i * config.batch_size
            batch_end = min((batch_i + 1) * config.batch_size, len(shuffled_dataset))
            batch_rows = shuffled_dataset.select(range(batch_start, batch_end)) 

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
                    content = parsed_msg["content"]
                    group_texts.append(content)

                # --- Reference Forward Pass---
                ref_logprobs_sums = []
                for seq_tokens, ob_len in zip(group_tokens, group_ob_lens):
                    ref_model_input = types.ModelInput.from_ints(tokens=seq_tokens)
                    logprobs_list = ref_client.compute_logprobs(ref_model_input).result()
                    
                    prompt_len = len(prompt_tokens)
                    response_logprobs = logprobs_list[prompt_len:]
                    ref_sum = sum(response_logprobs)
                    
                    ref_logprobs_sums.append(ref_sum)

                # Rewards
                pids_expanded = [pid] * len(group_texts)
                gts_expanded = [ground_truth] * len(group_texts)

                base_rewards = tracker.calculate_rewards(pids_expanded, group_texts, gts_expanded)
                
                final_rewards = []
                for i in range(len(base_rewards)):
                    sampled_sum = sum(group_logprobs[i])
                    ref_sum = ref_logprobs_sums[i]

                    kl_val = sampled_sum - ref_sum
                    
                    penalized = base_rewards[i] - (config.beta * kl_val)
                    final_rewards.append(penalized)
                    batch_kl_log.append(kl_val)
                
                batch_rewards_log.extend(final_rewards)
                
                # GRPO Normalization
                rewards_t = torch.tensor(final_rewards, dtype=torch.float32)
                mean, std = rewards_t.mean(), rewards_t.std(unbiased=False) + 1e-8
                advantages = ((rewards_t - mean) / std).tolist()

                for tokens, logprob, adv, ob_len in zip(
                    group_tokens, group_logprobs, advantages, group_ob_lens
                ):
                    input_tokens = tokens[:-1]
                    target_tokens = tokens[1:]
                    all_logprobs = [0.0] * ob_len + logprob
                    all_advantages = [0.0] * ob_len + [adv] * (len(input_tokens) - ob_len)
                    
                    datum = types.Datum(
                        model_input=types.ModelInput.from_ints(tokens=input_tokens),
                        loss_fn_inputs={
                            "target_tokens": target_tokens,
                            "logprobs": all_logprobs,
                            "advantages": all_advantages,
                        },
                    )
                    batch_datums.append(datum)
                
                tracker.update_batch_history(pids_expanded, group_texts, gts_expanded)

            # --- Accumulate Gradients ---
            _ = training_client.forward_backward(batch_datums, "importance_sampling").result()
            
            avg_reward = sum(batch_rewards_log)/len(batch_rewards_log)
            avg_kl = sum(batch_kl_log)/len(batch_kl_log)
            accum_metrics["reward"].append(avg_reward)
            accum_metrics["kl"].append(avg_kl)

            if ((batch_i + 1) % config.gradient_accumulation_steps == 0) or ((batch_i + 1) == n_batches_per_epoch):
                progress = global_step / total_optimizer_steps
                current_lr = max(0.0, config.learning_rate * (1.0 - progress))
                
                # Create new AdamParams object for each step
                current_adam_params = types.AdamParams(
                    learning_rate=current_lr, 
                    beta1=0.9, 
                    beta2=0.95, 
                    eps=1e-8
                )
                
                _ = training_client.optim_step(adam_params).result()
                
                final_r = sum(accum_metrics["reward"])/len(accum_metrics["reward"])
                final_k = sum(accum_metrics["kl"])/len(accum_metrics["kl"])
                
                metrics = {
                    "progress/global_step": global_step,
                    "progress/epoch": epoch + 1,
                    "optim/lr": current_lr,
                    "reward/mean": final_reward,
                    "metrics/kl": final_kl
                }
                ml_logger.log_metrics(metrics, step=global_step)
                logger.info(f"Step {global_step} | LR: {current_lr:.2e} | Reward: {final_reward:.4f} | KL: {final_kl:.4f}")
                
                accum_metrics = {"reward": [], "kl": []}
                global_step += 1

    checkpoint_utils.save_checkpoint(
        training_client=training_client, 
        name="final", 
        log_path=config.log_path, 
        kind="both", 
        loop_state={"batch": global_step}
    )
    tracker.save_state(os.path.join(config.log_path, "tracker_final.pkl"))
    ml_logger.close()
    logger.info("Training completed")

if __name__ == "__main__":
    chz.nested_entrypoint(main)