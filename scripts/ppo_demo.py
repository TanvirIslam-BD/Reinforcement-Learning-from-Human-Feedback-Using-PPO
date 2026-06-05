"""End-to-end TINY PPO/RLHF demo (CPU-friendly, but still slow).

  1. Load GPT-2 (actor) + a frozen reference + the sentiment judge
  2. Show a sample reward BEFORE training (what the judge thinks now)
  3. Run a couple of PPO steps (watch mean reward / loss)
  4. Show the actor's text vs the reference's text on a fixed prompt

Run:
    python scripts/ppo_demo.py
"""

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
from ppo_rlhf.evaluate import generate_some_text  # noqa: E402
from ppo_rlhf.model import load_ppo_models  # noqa: E402
from ppo_rlhf.reward import build_sentiment_pipeline, compute_rewards  # noqa: E402
from ppo_rlhf.train import (  # noqa: E402
    build_ppo_config,
    build_ppo_trainer,
    run_ppo_training,
)

BATCH_SIZE = 8     # tiny batch so a PPO step is feasible on CPU
LIMIT = 256        # tiny dataset subset
STEPS = 2          # just enough to see the loop run
SENTIMENT = "POSITIVE"
PROMPT = "Once upon a time in a land far"


def banner(text: str) -> None:
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def main() -> None:
    device = get_device()
    banner(f"Device: {device}  (CPU is expected; PPO will be slow)")

    model_cfg = ModelConfig()
    model, ref_model, tokenizer = load_ppo_models(model_cfg, device)
    sentiment_pipe = build_sentiment_pipeline(model_cfg, device)

    dataset = build_dataset(tokenizer, DataConfig(), model_cfg, limit=LIMIT)
    print(f"Dataset size: {len(dataset)}  |  batch_size: {BATCH_SIZE}")

    banner("BEFORE — reward the judge gives the reference model")
    ref_text = generate_some_text(PROMPT, ref_model, tokenizer, device=device)
    print(f"prompt   : {PROMPT}")
    print(f"ref text : {ref_text}")
    r = compute_rewards(sentiment_pipe, [PROMPT + ref_text], sentiment=SENTIMENT)[0]
    print(f">>> {SENTIMENT} reward (before): {r.item():.4f}")

    banner(f"TRAINING — {STEPS} PPO steps toward {SENTIMENT}")
    settings = PPOSettings(batch_size=BATCH_SIZE, mini_batch_size=BATCH_SIZE)
    ppo_config = build_ppo_config(settings, model_cfg)
    ppo_trainer = build_ppo_trainer(
        ppo_config, model, ref_model, tokenizer, dataset, collator
    )
    all_stats = run_ppo_training(
        ppo_trainer, tokenizer, sentiment_pipe,
        settings=settings, sentiment=SENTIMENT, max_steps=STEPS,
    )

    banner("AFTER — same prompt, the PPO-trained actor")
    new_text = generate_some_text(PROMPT, model, tokenizer, device=device)
    print(f"actor text : {new_text}")
    r2 = compute_rewards(sentiment_pipe, [PROMPT + new_text], sentiment=SENTIMENT)[0]
    print(f">>> {SENTIMENT} reward (after): {r2.item():.4f}")

    banner("RESULT")
    rewards = [round(s["ppo/mean_scores"], 4) for s in all_stats]
    losses = [round(s["ppo/loss/total"], 4) for s in all_stats]
    print(f"Mean reward per step: {rewards}")
    print(f"Total loss per step : {losses}")
    print("\n(Two steps won't visibly change the model — this just proves the")
    print(" Generate -> Score -> Learn loop runs. Real training needs a GPU.)")


if __name__ == "__main__":
    main()
