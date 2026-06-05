"""PPOTrainer setup and the core RLHF training loop.

The whole algorithm is three steps repeated over batches:
    1. GENERATE  — actor writes a continuation for each query.
    2. SCORE     — sentiment judge rates query+response  -> reward.
    3. LEARN     — ppo_trainer.step nudges the actor toward higher reward
                   (clipped policy update + KL penalty vs the frozen ref model).
"""

from __future__ import annotations

import torch
from tqdm import tqdm
from trl import PPOConfig, PPOTrainer
from trl.core import LengthSampler

from .config import GENERATION_KWARGS, ModelConfig, PPOSettings
from .reward import compute_rewards


def build_ppo_config(
    settings: PPOSettings | None = None,
    model_cfg: ModelConfig | None = None,
) -> PPOConfig:
    """Translate our dataclasses into TRL's PPOConfig."""
    settings = settings or PPOSettings()
    model_cfg = model_cfg or ModelConfig()
    return PPOConfig(
        model_name=model_cfg.model_name,
        learning_rate=settings.learning_rate,
        batch_size=settings.batch_size,
        mini_batch_size=settings.mini_batch_size,
        ppo_epochs=settings.ppo_epochs,
    )


def build_ppo_trainer(config, model, ref_model, tokenizer, dataset, data_collator):
    """Wire the four PPO characters together (actor, ref, judge-input, data)."""
    return PPOTrainer(
        config,
        model,
        ref_model,
        tokenizer,
        dataset=dataset,
        data_collator=data_collator,
    )


def run_ppo_training(
    ppo_trainer: PPOTrainer,
    tokenizer,
    sentiment_pipe,
    settings: PPOSettings | None = None,
    sentiment: str = "POSITIVE",
    max_steps: int | None = None,
    log: bool = True,
):
    """Run the Generate -> Score -> Learn loop over the trainer's dataloader.

    Returns the list of per-step PPO stat dicts.
    """
    settings = settings or PPOSettings()
    output_length_sampler = LengthSampler(
        settings.output_min_length, settings.output_max_length
    )
    gen_kwargs = dict(GENERATION_KWARGS)
    gen_kwargs["pad_token_id"] = tokenizer.eos_token_id

    all_stats = []
    for step, batch in enumerate(tqdm(ppo_trainer.dataloader, desc="PPO")):
        if max_steps is not None and step >= max_steps:
            break

        query_tensors = batch["input_ids"]

        # 1. GENERATE a response for every query.
        response_tensors = []
        for query in query_tensors:
            gen_len = output_length_sampler()
            gen_kwargs["max_new_tokens"] = gen_len
            response = ppo_trainer.generate(query, **gen_kwargs)
            response_tensors.append(response.squeeze()[-gen_len:])
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        # 2. SCORE query+response with the sentiment judge.
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        rewards = compute_rewards(sentiment_pipe, texts, sentiment=sentiment)

        # 3. LEARN — one PPO update step.
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        if log:
            ppo_trainer.log_stats(stats, batch, rewards)
        all_stats.append(stats)

    return all_stats


def save_ppo_model(model, tokenizer, model_dir: str) -> None:
    """Persist the trained policy + tokenizer to a directory."""
    import os

    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"Model saved to {model_dir}")
