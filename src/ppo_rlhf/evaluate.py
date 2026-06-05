"""Generate text and compare a PPO-trained model against a reference model.

Used to demonstrate the payoff: after PPO, the actor's continuations should
score higher on the target sentiment than the untrained reference model's.
"""

from __future__ import annotations

import pandas as pd
import torch
from trl.core import LengthSampler

from .config import SENT_KWARGS, get_device


def _gen_kwargs(tokenizer) -> dict:
    return {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }


def generate_some_text(
    input_text: str,
    model,
    tokenizer,
    max_new_tokens: int = 20,
    device: torch.device | None = None,
) -> str:
    """Continue `input_text` with `model` and return the decoded string."""
    device = device or get_device()
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    gen_kwargs = _gen_kwargs(tokenizer)
    gen_kwargs["max_new_tokens"] = max_new_tokens
    generated_ids = model.generate(input_ids, **gen_kwargs)
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


def compare_models_on_dataset(
    model,
    ref_model,
    dataset,
    tokenizer,
    sentiment_pipe,
    output_length_sampler: LengthSampler,
    sentiment_index: int = 1,
    bs: int = 16,
    device: torch.device | None = None,
    sent_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Sample `bs` queries; generate with both models; tabulate reward scores.

    `sentiment_index` selects which class score to report (1 == POSITIVE for
    the distilbert-imdb classifier, whose labels are [NEGATIVE, POSITIVE]).
    Returns a DataFrame with query, response before/after, and rewards.
    """
    device = device or get_device()
    sent_kwargs = sent_kwargs or SENT_KWARGS
    gen_kwargs = _gen_kwargs(tokenizer)

    game_data: dict = {}
    dataset.set_format("pandas")
    df_batch = dataset[:].sample(bs)
    game_data["query"] = df_batch["query"].tolist()
    query_tensors = df_batch["input_ids"].tolist()

    response_tensors_ref, response_tensors = [], []
    for i in range(bs):
        gen_len = output_length_sampler()
        query = torch.tensor(query_tensors[i]).unsqueeze(dim=0).to(device)

        out_ref = ref_model.generate(
            query, max_new_tokens=gen_len, **gen_kwargs
        ).squeeze()[-gen_len:]
        response_tensors_ref.append(out_ref)

        out = model.generate(
            query, max_new_tokens=gen_len, **gen_kwargs
        ).squeeze()[-gen_len:]
        response_tensors.append(out)

    game_data["response (before)"] = [
        tokenizer.decode(response_tensors_ref[i]) for i in range(bs)
    ]
    game_data["response (after)"] = [
        tokenizer.decode(response_tensors[i]) for i in range(bs)
    ]

    texts_before = [q + r for q, r in zip(game_data["query"], game_data["response (before)"])]
    game_data["rewards (before)"] = [
        out[sentiment_index]["score"] for out in sentiment_pipe(texts_before, **sent_kwargs)
    ]
    texts_after = [q + r for q, r in zip(game_data["query"], game_data["response (after)"])]
    game_data["rewards (after)"] = [
        out[sentiment_index]["score"] for out in sentiment_pipe(texts_after, **sent_kwargs)
    ]

    return pd.DataFrame(game_data)
