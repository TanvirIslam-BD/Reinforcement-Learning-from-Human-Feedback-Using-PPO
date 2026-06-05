"""RLHF with GPT-2 and PPO (TRL's PPOTrainer).

A modular, runnable adaptation of the IBM Skills Network "Reinforcement
Learning from Human Feedback Using PPO" lab. We fine-tune GPT-2 (lvwerra/
gpt2-imdb) with Proximal Policy Optimization so it writes more positive
(or negative) movie reviews, using a sentiment classifier as the reward.

This is the natural sequel to the reward_model package: there we *train* a
judge; here we *use* a judge to optimize a generator via RL.

Core loop (see ppo_rlhf.train.run_ppo_training):
    GENERATE responses -> SCORE with sentiment judge -> LEARN via PPO step.
"""

__version__ = "1.0.0"
