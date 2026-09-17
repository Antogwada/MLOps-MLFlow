"""Predict script for batch and single inference."""
import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config


def predict(input_csv: str, model_path: str = "models/best_model.joblib", output_csv: str = "predictions.csv"):
    """Exécute l'inférence sur un fichier CSV de clients."""
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Modèle introuvable : {model_file}. Exécutez make train d'abord.")

    model = joblib.load(model_file)
    df = pd.read_csv(input_csv)

    # Nettoyage TotalCharges si présent
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    predictions = model.predict(df)
    probabilities = model.predict_proba(df)[:, 1]

    result_df = df.copy()
    result_df["Predicted_Churn"] = predictions
    result_df["Churn_Probability"] = probabilities

    result_df.to_csv(output_csv, index=False)
    print(f"Prédictions sauvegardées dans : {output_csv} ({len(result_df)} clients traités)")
    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch inference")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--model", default="models/best_model.joblib", help="Model joblib path")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV path")
    args = parser.parse_args()

    predict(args.input, args.model, args.output)
