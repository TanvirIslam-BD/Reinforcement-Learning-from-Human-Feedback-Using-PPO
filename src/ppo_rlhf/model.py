"""Model + tokenizer setup for PPO training.

PPO needs THREE pieces:
    * model      — the "actor" (policy) we train. A causal LM with a value head.
    * ref_model  — a frozen copy of the actor; the KL penalty against it keeps
                   generations realistic instead of degenerate reward-hacking.
    * tokenizer  — GPT-2 has no pad token, so we reuse EOS.

The value head (AutoModelForCausalLMWithValueHead) adds a small regression head
that estimates "how good is this state" — PPO needs it to compute advantages.
"""

from __future__ import annotations

import torch
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

from .config import ModelConfig, get_device


def load_ppo_models(
    cfg: ModelConfig | None = None,
    device: torch.device | None = None,
    with_ref: bool = True,
):
    """Load (model, ref_model, tokenizer) for PPO.

    When `with_ref` is False, ref_model is None (PPOTrainer will then build its
    own reference internally).
    """
    cfg = cfg or ModelConfig()
    device = device or get_device()

    model = AutoModelForCausalLMWithValueHead.from_pretrained(cfg.model_name)
    ref_model = (
        AutoModelForCausalLMWithValueHead.from_pretrained(cfg.model_name)
        if with_ref
        else None
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    if ref_model is not None:
        ref_model.to(device)

    return model, ref_model, tokenizer


def load_trained_ppo_model(
    model_dir: str,
    device: torch.device | None = None,
):
    """Load a previously PPO-trained model + tokenizer from a directory."""
    device = device or get_device()
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
