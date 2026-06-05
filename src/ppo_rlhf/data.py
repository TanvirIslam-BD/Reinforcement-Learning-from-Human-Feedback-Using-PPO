"""IMDb dataset loading, filtering, tokenizing and the PPO collator.

Each example becomes a short *query* (the first few tokens of a real review)
that the model will continue. PPOTrainer feeds these queries in, the model
generates a continuation, and the sentiment judge scores query+response.

Pipeline:
    load imdb -> rename text->review -> drop short reviews
              -> truncate to a random LengthSampler length -> tokenize
"""

from __future__ import annotations

from datasets import load_dataset
from transformers import AutoTokenizer
from trl.core import LengthSampler

from .config import DATASET_NAME, DataConfig, ModelConfig


def build_dataset(
    tokenizer=None,
    data_cfg: DataConfig | None = None,
    model_cfg: ModelConfig | None = None,
    split: str = "train",
    limit: int | None = None,
):
    """Build the tokenized dataset PPOTrainer consumes.

    Returns a `datasets.Dataset` (torch-formatted) where each row has:
        review     — full original text
        input_ids  — truncated token ids (the query prompt)
        query      — decoded text of input_ids (needed by the reward judge)
    """
    data_cfg = data_cfg or DataConfig()
    model_cfg = model_cfg or ModelConfig()

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name)
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset(DATASET_NAME, split=split)
    ds = ds.rename_columns({"text": "review"})

    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    # Keep only reviews with enough text to form varied prompts.
    ds = ds.filter(
        lambda x: len(x["review"]) > data_cfg.min_review_chars, batched=False
    )

    # Random prompt length per sample for robustness to varied input sizes.
    input_size = LengthSampler(
        data_cfg.input_min_text_length, data_cfg.input_max_text_length
    )

    def tokenize(sample: dict) -> dict:
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


def collator(data: list[dict]) -> dict:
    """Group a list of samples into a dict of lists (PPOTrainer's format).

    [{"input_ids": a, "query": x}, {"input_ids": b, "query": y}]
      -> {"input_ids": [a, b], "query": [x, y]}
    """
    return dict((key, [d[key] for d in data]) for key in data[0])
