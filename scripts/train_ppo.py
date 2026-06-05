"""Train GPT-2 with PPO to write positive (or negative) reviews.

Examples:
    # Tiny CPU-friendly run (a handful of small steps)
    python scripts/train_ppo.py --tiny --steps 2

    # Train a "Happy LLM" (needs a GPU to be practical)
    python scripts/train_ppo.py --sentiment POSITIVE --save-dir outputs/ppo-good

    # Train a "Pessimistic LLM"
    python scripts/train_ppo.py --sentiment NEGATIVE --save-dir outputs/ppo-bad

Note: PPO on CPU is very slow. Use --tiny for a smoke test; use a GPU for real
training.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ppo_rlhf.config import (  # noqa: E402
    DataConfig,
    ModelConfig,
    PPOSettings,
    get_device,
)
from ppo_rlhf.data import build_dataset, collator  # noqa: E402
from ppo_rlhf.model import load_ppo_models  # noqa: E402
from ppo_rlhf.reward import build_sentiment_pipeline  # noqa: E402
from ppo_rlhf.train import (  # noqa: E402
    build_ppo_config,
    build_ppo_trainer,
    run_ppo_training,
    save_ppo_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GPT-2 with PPO (RLHF)")
    parser.add_argument("--sentiment", choices=["POSITIVE", "NEGATIVE"], default="POSITIVE")
    parser.add_argument("--steps", type=int, default=None,
                        help="Max PPO steps (default: full epoch over dataloader).")
    parser.add_argument("--batch-size", type=int, default=PPOSettings().batch_size)
    parser.add_argument("--lr", type=float, default=PPOSettings().learning_rate)
    parser.add_argument("--tiny", action="store_true",
                        help="Small dataset + small batch for a CPU smoke test.")
    parser.add_argument("--save-dir", default=str(Path("outputs") / "ppo-good"))
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}  |  sentiment target: {args.sentiment}")

    model_cfg = ModelConfig()
    batch_size = 8 if args.tiny else args.batch_size
    settings = PPOSettings(
        learning_rate=args.lr,
        batch_size=batch_size,
        mini_batch_size=batch_size,
    )

    model, ref_model, tokenizer = load_ppo_models(model_cfg, device)

    limit = 256 if args.tiny else None
    dataset = build_dataset(tokenizer, DataConfig(), model_cfg, limit=limit)
    print(f"Dataset size: {len(dataset)}  |  batch_size: {batch_size}")

    ppo_config = build_ppo_config(settings, model_cfg)
    ppo_trainer = build_ppo_trainer(
        ppo_config, model, ref_model, tokenizer, dataset, collator
    )
    sentiment_pipe = build_sentiment_pipeline(model_cfg, device)

    max_steps = args.steps if args.steps is not None else (2 if args.tiny else None)
    print(f"Training (max_steps={max_steps})...")
    all_stats = run_ppo_training(
        ppo_trainer, tokenizer, sentiment_pipe,
        settings=settings, sentiment=args.sentiment, max_steps=max_steps,
    )

    if all_stats:
        last = all_stats[-1]
        print(f"\nFinal mean reward: {last.get('ppo/mean_scores')}")
        print(f"Final total loss : {last.get('ppo/loss/total')}")

    save_ppo_model(model, tokenizer, args.save_dir)


if __name__ == "__main__":
    main()
