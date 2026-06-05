"""Small helpers from the lab: JSON I/O, batch padding, and stat printing."""

from __future__ import annotations

import json

import torch


def save_to_json(data: dict, file_path: str) -> None:
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Data successfully saved to {file_path}")


def load_from_json(file_path: str) -> dict:
    with open(file_path, "r") as f:
        return json.load(f)


def pad_sequence_to_length(tensor: torch.Tensor, length: int, pad_token_id: int) -> torch.Tensor:
    padding_length = length - tensor.size(0)
    if padding_length > 0:
        padding = torch.full(
            (padding_length,), pad_token_id, dtype=torch.long, device=tensor.device
        )
        return torch.cat((tensor, padding))
    return tensor


def pad_list_to_batch_size(
    tensors: list[torch.Tensor], batch_size: int, pad_token_id: int
) -> list[torch.Tensor]:
    """Pad each tensor to a common length and pad the list up to batch_size.

    PPOTrainer.step requires exactly `batch_size` items; when a manual batch is
    smaller (e.g. the step-by-step walkthrough) we top it up with pad-only rows.
    """
    max_length = max(t.size(0) for t in tensors)
    padded = [pad_sequence_to_length(t, max_length, pad_token_id) for t in tensors]
    while len(padded) < batch_size:
        padded.append(
            torch.full((max_length,), pad_token_id, dtype=torch.long, device=tensors[0].device)
        )
    return padded[:batch_size]


def print_ppo_stats(stats: dict, related_to_objective: bool = False) -> None:
    """Pretty-print the most useful keys from a PPO step's stats dict."""
    print("PPO Training Statistics\n")

    if related_to_objective:
        print("Objective Statistics:")
        print(f"  KL Divergence (objective/kl): {stats['objective/kl']}")
        print(f"  KL Coefficient (objective/kl_coef): {stats['objective/kl_coef']}")
        print(f"  Entropy (objective/entropy): {stats['objective/entropy']}\n")

        print("PPO Losses:")
        print(f"  Policy Loss (ppo/loss/policy): {stats['ppo/loss/policy']}")
        print(f"  Value Loss (ppo/loss/value): {stats['ppo/loss/value']}")
        print(f"  Total Loss (ppo/loss/total): {stats['ppo/loss/total']}\n")

        print("PPO Policy Statistics:")
        print(f"  Policy Entropy (ppo/policy/entropy): {stats['ppo/policy/entropy']}")
        print(f"  Approx KL (ppo/policy/approxkl): {stats['ppo/policy/approxkl']}")
        print(f"  Clip Fraction (ppo/policy/clipfrac): {stats['ppo/policy/clipfrac']}\n")
    else:
        print("Reward and Value Function Estimation:")
        print(f"  Mean Non-Score Reward (ppo/mean_non_score_reward): {stats['ppo/mean_non_score_reward']}")
        print(f"  Mean Scores (ppo/mean_scores): {stats['ppo/mean_scores']}")
        print(f"  Std Scores (ppo/std_scores): {stats['ppo/std_scores']}")
        print(f"  Value Prediction Error (ppo/val/error): {stats['ppo/val/error']}")
        print(f"  Explained Variance (ppo/val/var_explained): {stats['ppo/val/var_explained']}\n")

    print("Token Lengths:")
    print(f"  Queries Length Mean (tokens/queries_len_mean): {stats['tokens/queries_len_mean']}")
    print(f"  Responses Length Mean (tokens/responses_len_mean): {stats['tokens/responses_len_mean']}\n")

    print("Time Statistics:")
    print(f"  Total Time (time/ppo/total): {stats['time/ppo/total']} seconds\n")
