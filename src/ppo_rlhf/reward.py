"""The reward function: a sentiment classifier acts as the "judge".

For each generated text we run a sentiment pipeline and use the raw score of
the target class as the reward:
    * sentiment="POSITIVE"  -> trains a cheerful "Happy LLM"
    * sentiment="NEGATIVE"  -> trains a gloomy "Pessimistic LLM"

This is not a learned reward model, but it gives PPO a clean, automatic signal.
"""

from __future__ import annotations

import torch
from transformers import pipeline

from .config import SENT_KWARGS, ModelConfig, pipeline_device


def build_sentiment_pipeline(
    cfg: ModelConfig | None = None,
    device: torch.device | None = None,
):
    """Create the sentiment-analysis pipeline used as the reward judge."""
    cfg = cfg or ModelConfig()
    return pipeline(
        "sentiment-analysis",
        model=cfg.reward_model_name,
        device=pipeline_device(device),
    )


def compute_rewards(
    sentiment_pipe,
    texts: list[str],
    sentiment: str = "POSITIVE",
    sent_kwargs: dict | None = None,
) -> list[torch.Tensor]:
    """Score each text and return the target-class reward as a scalar tensor.

    `sentiment_pipe(texts, top_k=None, ...)` returns, per text, a list of
    {"label": ..., "score": ...} for every class. We pluck the score whose
    label matches `sentiment`.
    """
    sent_kwargs = sent_kwargs or SENT_KWARGS
    pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    scores = [
        item["score"]
        for output in pipe_outputs
        for item in output
        if item["label"] == sentiment
    ]
    return [torch.tensor(s) for s in scores]
