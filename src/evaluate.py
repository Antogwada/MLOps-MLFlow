"""Evaluation script computing test metrics and logging artifacts to MLflow."""
import os
import sys
import argparse
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import mlflow
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.utils import plot_roc_curve, plot_pr_curve, plot_confusion_matrix


def evaluate_model(config_path: str = "configs/config.yaml"):
    """Évalue le modèle sur le jeu de test et enregistre les artefacts dans MLflow."""
    cfg = get_config(config_path)

    # MLflow
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or cfg.mlflow.tracking_uri
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME") or cfg.mlflow.experiment_name
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    test_path = Path(cfg.data.processed_dir) / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"Données de test introuvables : {test_path}")

    # Charger le modèle
    local_model_path = Path("models/best_model.joblib")
    if local_model_path.exists():
        model = joblib.load(local_model_path)
        print(f"Modèle chargé depuis : {local_model_path}")
    else:
        # Tenter depuis le registre MLflow
        model_uri = f"models:/{cfg.mlflow.registered_model_name}/latest"
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Modèle chargé depuis le registre MLflow : {model_uri}")

    test_df = pd.read_csv(test_path)
    target_col = cfg.data.target
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    # Inférence
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calcul des métriques
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print("=== ÉVALUATION FINALE SUR LE TEST SET ===")
    print(f"Accuracy  : {acc * 100:.2f}%")
    print(f"Precision : {prec * 100:.2f}%")
    print(f"Recall    : {rec * 100:.2f}%")
    print(f"F1-Score  : {f1 * 100:.2f}%")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print("\nRapport détaillé :\n", classification_report(y_test, y_pred))

    # Génération des artefacts graphiques
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    roc_path = artifacts_dir / "roc_curve.png"
    pr_path = artifacts_dir / "pr_curve.png"
    cm_path = artifacts_dir / "confusion_matrix.png"
    preds_path = artifacts_dir / "test_predictions.csv"

    plot_roc_curve(y_test, y_prob, output_path=str(roc_path))
    plot_pr_curve(y_test, y_prob, output_path=str(pr_path))
    plot_confusion_matrix(y_test, y_pred, output_path=str(cm_path))

    # Sauvegarde des prédictions pour analyse d'erreurs
    pred_analysis_df = X_test.copy()
    pred_analysis_df["Actual_Churn"] = y_test
    pred_analysis_df["Predicted_Churn"] = y_pred
    pred_analysis_df["Churn_Probability"] = y_prob
    pred_analysis_df.to_csv(preds_path, index=False)

    # Logging dans un run d'évaluation MLflow
    with mlflow.start_run(run_name="final_test_evaluation"):
        mlflow.log_metrics({
            "test_accuracy": acc,
            "test_precision": prec,
            "test_recall": rec,
            "test_f1": f1,
            "test_roc_auc": roc_auc
        })
        mlflow.log_artifacts(str(artifacts_dir), artifact_path="evaluation_plots")
        print("Artefacts et métriques enregistrés avec succès dans MLflow !")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model and log artifacts")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    evaluate_model(args.config)
