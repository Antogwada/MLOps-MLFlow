"""Training script with MLflow tracking, GridSearchCV, and model registration."""
import os
import sys
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import mlflow
import mlflow.sklearn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.pipeline import build_pipeline


def train_model(config_path: str = "configs/config.yaml"):
    """Entraîne le modèle avec GridSearchCV, autologging MLflow et enregistrement."""
    cfg = get_config(config_path)

    # 1. Configuration MLflow
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or cfg.mlflow.tracking_uri
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME") or cfg.mlflow.experiment_name

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    print(f"MLflow Tracking URI : {tracking_uri}")
    print(f"MLflow Experiment   : {experiment_name}")

    # 2. Chargement des données
    train_path = Path(cfg.data.processed_dir) / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Données train introuvables : {train_path}. Exécutez preprocess d'abord.")

    train_df = pd.read_csv(train_path)
    target_col = cfg.data.target
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    # 3. Construction du pipeline et de la validation croisée
    base_pipeline = build_pipeline(cfg)
    cv_strategy = StratifiedKFold(n_splits=cfg.cv.n_splits, shuffle=True, random_state=cfg.data.random_state)

    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=cfg.model.param_grid,
        cv=cv_strategy,
        scoring=cfg.cv.scoring,
        n_jobs=-1,
        refit=True,
        verbose=1
    )

    # 4. Activer l'autologging MLflow
    mlflow.sklearn.autolog(
        log_input_examples=True,
        log_model_signatures=True,
        log_models=True,
        silent=False
    )

    with mlflow.start_run(run_name="gridsearch_cv_tuning") as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID : {run_id}")

        print(f"Lancement du GridSearchCV avec {cfg.cv.n_splits} folds (scoring: {cfg.cv.scoring})...")
        grid_search.fit(X_train, y_train)

        best_score = grid_search.best_score_
        best_params = grid_search.best_params_
        print(f"=== RÉSULTATS DU TUNING ===")
        print(f"Meilleur score ({cfg.cv.scoring}) : {best_score:.4f}")
        print(f"Meilleurs hyperparamètres : {best_params}")

        mlflow.log_metric("best_cv_score", best_score)
        mlflow.log_params({"best_" + k: v for k, v in best_params.items()})

        # Sauvegarde du modèle dans le registre local / MLflow Registry
        model_name = cfg.mlflow.registered_model_name
        try:
            print(f"Enregistrement du modèle dans le MLflow Model Registry sous '{model_name}'...")
            model_uri = f"runs:/{run_id}/best_estimator"
            mlflow.register_model(model_uri=model_uri, name=model_name)
            print(f"Modèle enregistré avec succès : {model_name}")
        except Exception as e:
            print(f"Note sur l'enregistrement du modèle : {e}")

        # Sauvegarder également le meilleur modèle en local dans models/
        out_model_dir = Path("models")
        out_model_dir.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump(grid_search.best_estimator_, out_model_dir / "best_model.joblib")
        print(f"Meilleur modèle sauvegardé localement dans : {out_model_dir / 'best_model.joblib'}")

    return grid_search.best_estimator_


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model with MLflow tracking")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    train_model(args.config)
