import logging
import time
import os
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
from utils import MAX_TRAIN_SET_SIZE

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)

# Suppress tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@chz.chz
class Config:
    # Added your specific args
    train_dataset: str = "datasets/train_samples_dsr_2000.json"
    tinker_api_key: str | None = None

    base_url: str | None = None
    log_path: str = "/tmp/tinker-experiments/grpo-math"
    model_name: str = "Qwen/Qwen3-8B-Base"

    batch_size: int = 4
    group_size: int = 4
    learning_rate: float = 1e-5
    lora_rank: int = 32
    save_every: int = 10
    max_tokens: int = 800

    # Reward Hyperparams
    w_lr: float = 1.0
    type_lr: str = "cosine"
    temperature: float = 0.7
    mode: str = "min" # 'min' or 'mean'
    rep_ngram_size: int = 3
    rep_penalty: float = 0.0


def main(config: Config):
    # Set up API key
    if config.tinker_api_key:
        os.environ["TINKER_API_KEY"] = config.tinker_api_key

    # Setup logging
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=None,
        wandb_name=None,
        config=config,
        do_configure_logging_module=True,
    )

    # Get tokenizer and renderer
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")

    # Initialize tracker
    tracker = HistoryTracker(
        tokenizer=tokenizer, 
        w_lr=config.w_lr, 
        type_lr=config.type_lr,
        rep_ngram_size=config.rep_ngram_size,
        rep_penalty=config.rep_penalty,
        mode=config.mode
    )

    # Load dataset
    logger.info("Loading dataset...")
    raw_dataset = datasets.load_dataset("json", data_files=config.train_dataset, split="train")

    # train_grpo.py uses 'idx' from the dataset enumeration
    # and shifts eval IDs by MAX_TRAIN_SET_SIZE. 
    # Since we are only doing train here, we just need the 0-based index.
    def add_prefix(example, idx):
        # Handle list vs string
        if isinstance(example["question"], list):
            q_str = " ".join(str(item) for item in example["question"])
        else:
            q_str = example["question"]
            
        # We just store the question string here; we'll render it in the loop
        return {
            "q_str": q_str,
            "ground_truth": example["ground_truth"],
            "prompt_idx": idx
        }
    train_dataset = raw_dataset.map(add_prefix, with_indices=True)
    n_train_batches = len(train_dataset) // config.batch_size

    # Setup training client
    service_client = tinker.ServiceClient(base_url=config.base_url)

    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    if resume_info:
        training_client = service_client.create_training_client_from_state(
            resume_info["state_path"]
        )
        start_batch = resume_info["batch"]
        logger.info(f"Resuming from batch {start_batch}")
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank
        )
        start_batch = 0

    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=config.temperature
    )
    # Optimizer step
    adam_params = types.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
    )

    logger.info(f"Training for {n_train_batches} batches")

    #  Main training loop
    for batch_idx in range(start_batch, n_train_batches):
        # Setup metrics for logging
        t_start = time.time()
        step = batch_idx
        metrics: dict[str, float] = {
            "progress/batch": batch_idx,
            "optim/lr": config.learning_rate,
            "progress/done_frac": (batch_idx + 1) / n_train_batches,
        }

        # Save checkpoint
        if step % config.save_every == 0 and step > 0:
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"{step:06d}",
                log_path=config.log_path,
                kind="state",
                loop_state={"batch": batch_idx},
            )

        # Get training batch and convert to datums online
        batch_start = batch_idx * config.batch_size
        batch_end = min((batch_idx + 1) * config.batch_size, len(train_dataset))
        batch_rows = train_dataset.select(range(batch_start, batch_end))

        # Sync weights
        sampling_path = training_client.save_weights_for_sampler(name=f"{step:06d}").result().path
        sampling_client = service_client.create_sampling_client(model_path=sampling_path)

        training_datums: list[types.Datum] = []
        batch_rewards_log: list[float] = []
        batch_futures: list[list[Future[types.SampleResponse]]] = []
        batch_prompts_tokens: list[list[int]] = []

        # Submit sampling requests
        for q_str in batch_rows["q_str"]:
            convo = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q_str}
            ]
            model_input = renderer.build_generation_prompt(convo)
            prompt_tokens = model_input.to_ints()

            sample_futures: list[Future[types.SampleResponse]] = []
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

        # Process results
        for sample_futures, prompt_tokens, ground_truth, pid in zip(
            batch_futures, batch_prompts_tokens, batch_rows["ground_truth"], batch_rows["prompt_idx"]
        ):
            group_tokens: list[list[int]] = []
            group_logprobs: list[list[float]] = []
            group_ob_lens: list[int] = []
            group_response_texts: list[str] = []

            # 1. Gather Results
            for future in sample_futures:
                sample_result = future.result()
                sampled_tokens = sample_result.sequences[0].tokens
                sampled_logprobs = sample_result.sequences[0].logprobs
                assert sampled_logprobs is not None

                # Reconstruct full sequence
                all_tokens = prompt_tokens + sampled_tokens
                group_tokens.append(all_tokens)
                group_ob_lens.append(len(prompt_tokens) - 1)
                group_logprobs.append(sampled_logprobs)
                
                # Parse response
                # This automatically strips special tokens (<|im_start|>, etc.)
                # returning a clean dictionary: {'role': 'assistant', 'content': '...'}
                parsed_msg, _ = renderer.parse_response(sampled_tokens)

                # We extract just the content string for the reward function
                if parsed_msg and "content" in parsed_msg:
                    response_text = parsed_msg["content"]
                else:
                    # Fallback if parsing fails (e.g. empty generation)
                    response_text = ""
                
                group_response_texts.append(response_text)

            # 2. Calculate Rewards
            # We pass the list of N responses for this single prompt ID
            # Tracker returns a list of N floats
            pids_expanded = [pid] * len(group_response_texts)
            gts_expanded = [ground_truth] * len(group_response_texts)
            
            group_rewards = tracker.calculate_rewards(
                pids_expanded, group_response_texts, gts_expanded
            )
            batch_rewards_log.append(sum(group_rewards) / len(group_rewards))

            # 3. Calculate Advantages (GRPO)
            rewards_tensor = torch.tensor(group_rewards, dtype=torch.float32)
            mean = rewards_tensor.mean()
            std = rewards_tensor.std(unbiased=False) + 1e-8
            advantages = ((rewards_tensor - mean) / std).tolist()

            # Skip if no variance
            if std < 1e-6:
                continue

            # 4. Construct Datums
            for tokens, logprob, advantage, ob_len in zip(
                group_tokens, group_logprobs, advantages, group_ob_lens
            ):
                input_tokens = tokens[:-1]
                target_tokens = tokens[1:]
                
                # Pad logprobs/advantages for the prompt portion
                all_logprobs = [0.0] * ob_len + logprob
                all_advantages = [0.0] * ob_len + [advantage] * (len(input_tokens) - ob_len)
                
                datum = types.Datum(
                    model_input=types.ModelInput.from_ints(tokens=input_tokens),
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                        "logprobs": TensorData.from_torch(torch.tensor(all_logprobs)),
                        "advantages": TensorData.from_torch(torch.tensor(all_advantages)),
                    },
                )
                training_datums.append(datum)

       # Training step
        if training_datums:
            fwd_bwd_future = training_client.forward_backward(
                training_datums, loss_fn="importance_sampling"
            )
            optim_step_future = training_client.optim_step(adam_params)
            
            _fwd_bwd_result = fwd_bwd_future.result()
            _optim_result = optim_step_future.result()

            # Log metrics
            metrics = {
                "progress/batch": batch_idx,
                "optim/lr": config.learning_rate,
                "time/total": time.time() - t_start,
                "reward/mean": sum(batch_rewards_log) / len(batch_rewards_log) if batch_rewards_log else 0.0
            }
            ml_logger.log_metrics(metrics, step=batch_idx)
            logger.info(f"Batch {batch_idx} | Reward: {metrics['reward/mean']:.4f}")
        else:
            logger.info(f"Batch {batch_idx} | Skipped (No gradients)")

    # Save final checkpoint
    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name="final",
        log_path=config.log_path,
        kind="both",
        loop_state={"batch": n_train_batches},
    )
    ml_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    chz.nested_entrypoint(main)