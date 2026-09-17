"""Data download module for Telco Customer Churn."""
import os
import sys
import argparse
import urllib.request
from pathlib import Path

# Assurer la visibilité des modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config

# URL directe et fiable du dataset Telco Customer Churn (IBM / Kaggle original)
DATASET_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"


def download_data(config_path: str = "configs/config.yaml"):
    """Télécharge le fichier CSV brut et le sauvegarde dans data/raw.csv."""
    cfg = get_config(config_path)
    out_path = Path(cfg.data.raw_csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Téléchargement du dataset Telco Churn depuis : {DATASET_URL}")
    urllib.request.urlretrieve(DATASET_URL, out_path)

    import pandas as pd
    df = pd.read_csv(out_path)
    print(f"Dataset sauvegardé avec succès dans : {out_path}")
    print(f"Dimensions : {df.shape[0]} lignes, {df.shape[1]} colonnes.")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Telco Customer Churn Dataset")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    download_data(args.config)
