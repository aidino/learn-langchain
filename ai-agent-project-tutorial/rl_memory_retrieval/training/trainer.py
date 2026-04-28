"""PPO training loop for the memory retrieval RL agent.

Uses stable-baselines3 PPO with configurable hyperparameters from Settings.
Wraps the MemoryRetrievalEnv in a DummyVecEnv for SB3 compatibility.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_memory_retrieval.config import Settings
from rl_memory_retrieval.training.environment import MemoryRetrievalEnv


def train_model(
    train_items: list[dict],
    config: Settings,
    seed: int = 42,
) -> PPO:
    """Train a PPO model on the memory retrieval environment.

    Args:
        train_items: List of training items (query + candidates dicts).
        config: Application settings with training hyperparameters.
        seed: Random seed for reproducibility.

    Returns:
        Trained PPO model.
    """
    top_k = config.training.top_k_candidates

    train_env = DummyVecEnv(
        [lambda: MemoryRetrievalEnv(train_items, top_k=top_k, seed=seed)]
    )

    tc = config.training
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=tc.learning_rate,
        n_steps=tc.n_steps,
        batch_size=tc.batch_size,
        gamma=tc.gamma,
        gae_lambda=tc.gae_lambda,
        ent_coef=tc.ent_coef,
        clip_range=tc.clip_range,
        verbose=0,
        seed=seed,
    )

    model.learn(total_timesteps=tc.total_timesteps)

    return model


def save_model(
    model: PPO,
    embeddings: np.ndarray,
    output_dir: str,
    metadata: dict | None = None,
) -> Path:
    """Save a trained model and its associated embeddings.

    Args:
        model: Trained PPO model.
        embeddings: Numpy array of chunk embeddings.
        output_dir: Directory to save artifacts to.
        metadata: Optional metadata dict to save alongside the model.

    Returns:
        Path to the output directory.
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    model.save(str(path / "ppo_model"))
    np.save(str(path / "embeddings.npy"), embeddings)

    if metadata is not None:
        import json

        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    return path


def load_model(model_dir: str) -> tuple[PPO, np.ndarray, dict | None]:
    """Load a previously saved model and embeddings.

    Args:
        model_dir: Directory containing saved model artifacts.

    Returns:
        Tuple of (PPO model, embeddings array, metadata dict or None).
    """
    path = Path(model_dir)

    model = PPO.load(str(path / "ppo_model"))
    embeddings = np.load(str(path / "embeddings.npy"))

    metadata = None
    metadata_path = path / "metadata.json"
    if metadata_path.exists():
        import json

        with open(metadata_path) as f:
            metadata = json.load(f)

    return model, embeddings, metadata
