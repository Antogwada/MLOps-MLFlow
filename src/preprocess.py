"""Preprocessing script for Telco Customer Churn."""
import os
import sys
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config


def preprocess_data(config_path: str = "configs/config.yaml"):
    """Nettoie le dataset Telco Churn, convertit les colonnes et sépare train/test."""
    cfg = get_config(config_path)
    raw_path = Path(cfg.data.raw_csv_path)

    if not raw_path.exists():
        raise FileNotFoundError(f"Fichier brut introuvable : {raw_path}. Exécutez get_data d'abord.")

    df = pd.read_csv(raw_path)
    print(f"Chargement de {raw_path} : {df.shape}")

    # 1. Nettoyage de TotalCharges (espaces vides convertis en NaN puis remplis avec mediane)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
    missing_tc = df["TotalCharges"].isna().sum()
    if missing_tc > 0:
        median_tc = df["TotalCharges"].median()
        df["TotalCharges"] = df["TotalCharges"].fillna(median_tc)
        print(f"Remplacement de {missing_tc} valeurs manquantes dans TotalCharges par la médiane ({median_tc:.2f})")

    # 2. Encodage de la variable cible Churn (Yes/No -> 1/0)
    target_col = cfg.data.target
    if df[target_col].dtype == object:
        df[target_col] = df[target_col].map({"Yes": 1, "No": 0})
    print(f"Distribution de la cible {target_col} :\n{df[target_col].value_counts(normalize=True)}")

    # 3. Sélection des colonnes utiles
    features_needed = cfg.features.numeric + cfg.features.categorical + [target_col]
    df = df[[col for col in features_needed if col in df.columns]]

    # 4. Séparation Train / Test stratifiée
    train_df, test_df = train_test_split(
        df,
        test_size=cfg.data.test_size,
        random_state=cfg.data.random_state,
        stratify=df[target_col]
    )

    out_dir = Path(cfg.data.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.csv"
    test_path = out_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Données prétraitées sauvegardées :")
    print(f"  Train: {train_path} ({len(train_df)} lignes)")
    print(f"  Test : {test_path} ({len(test_df)} lignes)")
    return train_path, test_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Telco Churn Dataset")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    preprocess_data(args.config)
