# Reinforcement Learning from Human Feedback Using PPO

Fine-tune GPT-2 with **Proximal Policy Optimization (PPO)** so it writes more
**positive** (or **negative**) movie reviews. A sentiment classifier plays the
role of the reward "judge".

This is the reinforcement-learning stage of the RLHF pipeline:

```
1. SFT / Instruction tuning   → teach the model to follow instructions
2. Reward modeling            → train a judge from human preferences
3. RLHF / PPO  (THIS)         → use a judge to improve the generator
```

> A modular, runnable adaptation of the IBM Skills Network lab *"Reinforcement
> Learning from Human Feedback Using PPO"* (based on TRL's `gpt2-sentiment` example).

## The mental model — 4 characters

| Character | In the code | Job |
|-----------|-------------|-----|
| 🎭 **Actor** (policy) | `model` (`AutoModelForCausalLMWithValueHead`) | Writes text. This is what we train. |
| 📏 **Reference** | `ref_model` (frozen copy) | A leash: the KL penalty keeps text realistic. |
| ⚖️ **Judge** (reward) | `sentiment_pipe` (`lvwerra/distilbert-imdb`) | Scores how positive/negative the text is. |
| 🧠 **Coach** | `ppo_trainer` (TRL `PPOTrainer`) | Turns the score into a policy update. |

## The core loop

Everything reduces to three steps, repeated over batches
(`ppo_rlhf.train.run_ppo_training`):

```
GENERATE  actor continues each prompt
SCORE     judge rates query+response  → reward
LEARN     ppo_trainer.step nudges the actor toward higher reward
```

Set `sentiment="POSITIVE"` for a "Happy LLM", `"NEGATIVE"` for a "Pessimistic LLM".

## How it works

1. **Data** — `imdb`: each review is truncated to a short random-length prompt
   (TRL's `LengthSampler`) the model must continue.
2. **Model** — `lvwerra/gpt2-imdb` wrapped with a **value head** (PPO needs it to
   estimate advantages); a frozen copy is the reference.
3. **Reward** — a sentiment pipeline returns the target-class score as the reward.
4. **Train** — `PPOTrainer.step(queries, responses, rewards)` does a clipped
   policy update + KL penalty vs the reference.
5. **Compare** — generate from the trained model vs the reference and tabulate
   the sentiment reward before/after.

## Project layout

```
src/ppo_rlhf/
  config.py     # paths, model names, data + PPO settings, generation kwargs
  data.py       # build_dataset (load/filter/LengthSampler-tokenize) + collator
  model.py      # load actor + frozen reference (value head), load trained model
  reward.py     # sentiment-pipeline judge + compute_rewards
  train.py      # PPOConfig/PPOTrainer wiring + the Generate→Score→Learn loop
  evaluate.py   # generate_some_text, compare_models_on_dataset
  utils.py      # JSON I/O, batch padding, print_ppo_stats
scripts/
  train_ppo.py    # train toward POSITIVE/NEGATIVE; --tiny for a CPU smoke test
  ppo_demo.py     # tiny before/after demo: prove the loop runs (CPU)
  compare_ppo.py  # tabulate reward of a saved PPO model vs the reference
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows PowerShell
# source .venv/bin/activate      # Linux/macOS

# Install PyTorch first (pick ONE)
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu   # CPU
# pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121  # CUDA

pip install -r requirements.txt
pip install -e .
```

## Usage

```bash
# Tiny CPU smoke test — proves the Generate→Score→Learn loop runs
python scripts/ppo_demo.py

# Train a "Happy LLM" (use a GPU — PPO on CPU is very slow)
python scripts/train_ppo.py --sentiment POSITIVE --save-dir outputs/ppo-good

# Train a "Pessimistic LLM"
python scripts/train_ppo.py --sentiment NEGATIVE --save-dir outputs/ppo-bad

# Tiny run (small dataset + small batch + 2 steps)
python scripts/train_ppo.py --tiny --steps 2

# Compare a saved model against the untrained reference
python scripts/compare_ppo.py --model-dir outputs/ppo-good --bs 16
```

## How this differs from the lab notebook

| Notebook | This project |
|----------|--------------|
| Training loop **commented out**; downloads a pretrained `.pkl`/`.tar.gz` | Real, runnable loop (`run_ppo_training`); `--tiny` for CPU |
| One long top-to-bottom notebook with a manual single-step walkthrough | Modular package: data / model / reward / train / evaluate |
| Hard-coded `batch_size=128`, manual padding to meet the minimum | `PPOSettings` (configurable); `--tiny` uses a small batch so a step fits on CPU |
| POSITIVE vs NEGATIVE handled by editing cells | `--sentiment {POSITIVE,NEGATIVE}` flag |

## Notes

- **CPU is very slow** for PPO (each step generates a whole batch then back-props).
  `ppo_demo.py` / `--tiny` keep it tractable for a smoke test only.
- **Value head**: `AutoModelForCausalLMWithValueHead` adds the critic PPO needs.
- **Pad token**: GPT-2 has none → `pad_token = eos_token`.
- **Pinned versions** match the lab (`trl 0.11`, `transformers 4.43.4`).

## Credits

Adapted from the IBM Skills Network lab by Joseph Santarcangelo, Ashutosh Sagar,
and Hailey Quach. Based on Hugging Face TRL's
[`gpt2-sentiment`](https://github.com/huggingface/trl/blob/main/examples/notebooks/gpt2-sentiment.ipynb)
example. Dataset: [imdb](https://huggingface.co/datasets/imdb).
