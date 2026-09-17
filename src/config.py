"""Configuration module for Telco Churn MLflow project."""
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any
import yaml


@dataclass
class DataConfig:
    raw_csv_path: str
    processed_dir: str
    target: str
    test_size: float
    random_state: int


@dataclass
class FeaturesConfig:
    numeric: List[str]
    categorical: List[str]


@dataclass
class ModelConfig:
    type: str
    param_grid: Dict[str, Any]


@dataclass
class CVConfig:
    strategy: str
    n_splits: int
    scoring: str


@dataclass
class MLflowConfig:
    experiment_name: str
    tracking_uri: str
    registered_model_name: str


@dataclass
class AppConfig:
    data: DataConfig
    features: FeaturesConfig
    model: ModelConfig
    cv: CVConfig
    mlflow: MLflowConfig

    @classmethod
    def from_yaml(cls, path: str = "configs/config.yaml") -> "AppConfig":
        config_path = Path(path)
        if not config_path.is_absolute():
            base_dir = Path(__file__).resolve().parent.parent
            config_path = base_dir / path

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration introuvable : {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        return cls(
            data=DataConfig(**raw["data"]),
            features=FeaturesConfig(**raw["features"]),
            model=ModelConfig(**raw["model"]),
            cv=CVConfig(**raw["cv"]),
            mlflow=MLflowConfig(**raw["mlflow"]),
        )


def get_config(path: str = "configs/config.yaml") -> AppConfig:
    return AppConfig.from_yaml(path)
