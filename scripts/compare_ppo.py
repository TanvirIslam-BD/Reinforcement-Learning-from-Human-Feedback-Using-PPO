"""Compare a PPO-trained model against the untrained reference model.

Generates responses from both on a sample of IMDb prompts and tabulates the
sentiment reward before (reference) vs after (PPO) training.

Run:
    python scripts/compare_ppo.py --model-dir outputs/ppo-good
    python scripts/compare_ppo.py --model-dir outputs/ppo-bad --bs 16
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trl.core import LengthSampler  # noqa: E402

from ppo_rlhf.config import (  # noqa: E402
    DataConfig,
    ModelConfig,
    PPOSettings,
    get_device,
)
from ppo_rlhf.data import build_dataset  # noqa: E402
from ppo_rlhf.evaluate import compare_models_on_dataset  # noqa: E402
from ppo_rlhf.model import load_ppo_models, load_trained_ppo_model  # noqa: E402
from ppo_rlhf.reward import build_sentiment_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PPO model vs reference")
    parser.add_argument("--model-dir", required=True,
                        help="Directory of a PPO-trained model (from train_ppo.py).")
    parser.add_argument("--bs", type=int, default=16, help="Number of prompts to sample.")
    parser.add_argument("--limit", type=int, default=512,
                        help="Dataset subset size to sample prompts from.")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    model_cfg = ModelConfig()
    # Reference (untrained) model + tokenizer.
    _, ref_model, tokenizer = load_ppo_models(model_cfg, device)
    # The PPO-trained model under test.
    model, _ = load_trained_ppo_model(args.model_dir, device)

    dataset = build_dataset(tokenizer, DataConfig(), model_cfg, limit=args.limit)
    sentiment_pipe = build_sentiment_pipeline(model_cfg, device)

    settings = PPOSettings()
    output_length_sampler = LengthSampler(
        settings.output_min_length, settings.output_max_length
    )

    df = compare_models_on_dataset(
        model, ref_model, dataset, tokenizer, sentiment_pipe,
        output_length_sampler, bs=args.bs, device=device,
    )

    print("\n=== Per-prompt results ===")
    print(df.to_string(index=False))
    print("\n=== Mean reward ===")
    print(f"before (reference): {df['rewards (before)'].mean():.4f}")
    print(f"after  (PPO)      : {df['rewards (after)'].mean():.4f}")


if __name__ == "__main__":
    main()
