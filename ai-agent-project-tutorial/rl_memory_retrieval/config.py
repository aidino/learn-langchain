"""Configuration system using pydantic-settings with YAML support."""

from pathlib import Path
from typing import Any, Tuple, Type

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class EmbeddingsConfig(BaseModel):
    """Embedding backend configuration."""

    backend: str = "openai"
    model: str = "text-embedding-3-small"
    batch_size: int = 64


class ExtractionConfig(BaseModel):
    """Optional metadata extraction configuration."""

    enabled: bool = False
    instruction: str = "Extract entities, topics, and key facts"


class ChunkingConfig(BaseModel):
    """Text chunking configuration."""

    chunk_size: int = 300
    chunk_overlap: int = 50


class QueryGenConfig(BaseModel):
    """LLM query generation configuration."""

    queries_per_chunk: int = 3
    model: str = "gpt-4o-mini"


class TrainingConfig(BaseModel):
    """PPO training hyperparameters."""

    total_timesteps: int = 12000
    learning_rate: float = 3e-4
    n_steps: int = 256
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    clip_range: float = 0.2
    top_k_candidates: int = 8
    train_split: float = 0.70
    val_split: float = 0.15


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""

    sample_size: int = 20
    use_llm_judge: bool = True


class Settings(BaseSettings):
    """Main application settings.

    Loads from (priority order):
    1. CLI arguments / init kwargs (highest)
    2. Environment variables (RL_ prefix)
    3. .env file
    4. config.yaml file
    5. Defaults (lowest)
    """

    model_config = SettingsConfigDict(
        env_prefix="RL_",
        env_file=".env",
        yaml_file="config.yaml",
        env_nested_delimiter="__",
    )

    source: str | None = None
    output_dir: str = "./model_output"

    embeddings: EmbeddingsConfig = EmbeddingsConfig()
    extraction: ExtractionConfig = ExtractionConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    query_generation: QueryGenConfig = QueryGenConfig()
    training: TrainingConfig = TrainingConfig()
    evaluation: EvaluationConfig = EvaluationConfig()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )
