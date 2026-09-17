"""Pipeline builder module with ColumnTransformer and classification model."""
import sys
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import AppConfig


def build_preprocessor(cfg: AppConfig) -> ColumnTransformer:
    """Construit le préprocesseur ColumnTransformer pour variables numériques et catégorielles."""
    # Pipeline pour variables numériques : Imputation médiane + Standardisation
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Pipeline pour variables catégorielles : Imputation mode + One-Hot Encoding
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, cfg.features.numeric),
        ("cat", categorical_transformer, cfg.features.categorical)
    ])

    return preprocessor


def build_pipeline(cfg: AppConfig) -> Pipeline:
    """Construit le pipeline scikit-learn complet (preprocessor + classifieur)."""
    preprocessor = build_preprocessor(cfg)

    # Sélection du modèle selon config
    if cfg.model.type == "random_forest":
        classifier = RandomForestClassifier(random_state=cfg.data.random_state)
    else:
        classifier = LogisticRegression(random_state=cfg.data.random_state, max_iter=500)

    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    return full_pipeline
