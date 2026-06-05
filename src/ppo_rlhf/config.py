"""Central configuration for the PPO / RLHF pipeline.

All tunable knobs live here. Values mirror the original Skills Network
"RLHF Using PPO" lab defaults; device is auto-detected.

Big picture: we nudge GPT-2 to write more *positive* (or *negative*) movie
reviews. A sentiment classifier is the "judge" (reward), and PPO is the
"coach" that updates the model toward higher-scoring text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Hugging Face dataset of 50k labelled movie reviews.
DATASET_NAME = "imdb"


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pipeline_device(device: torch.device | None = None) -> int:
    """transformers pipelines want an int: 0 for first GPU, -1 for CPU."""
    device = device or get_device()
    return 0 if device.type == "cuda" else -1


@dataclass
class ModelConfig:
    # GPT-2 already fine-tuned on IMDb — the "actor" we keep training.
    model_name: str = "lvwerra/gpt2-imdb"
    # DistilBERT sentiment classifier — the "judge" that hands out rewards.
    reward_model_name: str = "lvwerra/distilbert-imdb"


@dataclass
class DataConfig:
    # Keep only reviews longer than this many characters.
    min_review_chars: int = 200
    # Each review is truncated to a random prompt length in [min, max) tokens
    # (via TRL's LengthSampler) so the model sees varied input sizes.
    input_min_text_length: int = 2
    input_max_text_length: int = 8


@dataclass
class PPOSettings:
    learning_rate: float = 1.41e-5
    # batch_size = queries per PPO step; mini_batch_size must divide it.
    # The lab uses 128; the tiny demo overrides these for CPU.
    batch_size: int = 128
    mini_batch_size: int = 128
    ppo_epochs: int = 4
    # Lengths (in new tokens) sampled for each generated response.
    output_min_length: int = 4
    output_max_length: int = 16


# Sampling parameters passed to model.generate during rollouts.
# min_length=-1 / top_k=0 / top_p=1 / do_sample=True => free sampling.
GENERATION_KWARGS: dict = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": 50256,  # GPT-2 EOS token id
}

# Sentiment-pipeline parameters: return every label's raw (un-softmaxed)
# score so we can read the POSITIVE / NEGATIVE value directly.
SENT_KWARGS: dict = {"top_k": None, "function_to_apply": "none", "batch_size": 2}
