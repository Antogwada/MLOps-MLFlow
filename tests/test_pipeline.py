"""Sanity checks and unit tests for Telco Churn pipeline."""
import unittest
import pandas as pd
import numpy as np
from src.config import get_config
from src.pipeline import build_pipeline


class TestChurnPipeline(unittest.TestCase):
    """Vérifications du pipeline scikit-learn."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = get_config()
        cls.pipeline = build_pipeline(cls.cfg)

        # Création d'un mini-dataset de test synthétique avec les colonnes requises
        n_samples = 20
        np.random.seed(42)

        data = {
            "tenure": np.random.randint(1, 72, size=n_samples),
            "MonthlyCharges": np.random.uniform(20.0, 110.0, size=n_samples),
            "TotalCharges": np.random.uniform(20.0, 5000.0, size=n_samples),
            "gender": np.random.choice(["Male", "Female"], size=n_samples),
            "SeniorCitizen": np.random.choice([0, 1], size=n_samples),
            "Partner": np.random.choice(["Yes", "No"], size=n_samples),
            "Dependents": np.random.choice(["Yes", "No"], size=n_samples),
            "PhoneService": np.random.choice(["Yes", "No"], size=n_samples),
            "MultipleLines": np.random.choice(["Yes", "No", "No phone service"], size=n_samples),
            "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], size=n_samples),
            "OnlineSecurity": np.random.choice(["Yes", "No", "No internet service"], size=n_samples),
            "OnlineBackup": np.random.choice(["Yes", "No", "No internet service"], size=n_samples),
            "DeviceProtection": np.random.choice(["Yes", "No", "No internet service"], size=n_samples),
            "TechSupport": np.random.choice(["Yes", "No", "No internet service"], size=n_samples),
            "StreamingTV": np.random.choice(["Yes", "No", "No internet service"], size=n_samples),
            "StreamingMovies": np.random.choice(["Yes", "No", "No internet service"], size=n_samples),
            "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], size=n_samples),
            "PaperlessBilling": np.random.choice(["Yes", "No"], size=n_samples),
            "PaymentMethod": np.random.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], size=n_samples),
        }
        cls.X_dummy = pd.DataFrame(data)
        cls.y_dummy = np.random.choice([0, 1], size=n_samples)

    def test_pipeline_fit_and_predict(self):
        """Vérifie que le pipeline s'entraîne et prédit sans erreur sur données tabulaires."""
        self.pipeline.fit(self.X_dummy, self.y_dummy)
        preds = self.pipeline.predict(self.X_dummy)
        probs = self.pipeline.predict_proba(self.X_dummy)

        self.assertEqual(len(preds), len(self.X_dummy))
        self.assertEqual(probs.shape, (len(self.X_dummy), 2))
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

    def test_pipeline_handles_missing_values(self):
        """Vérifie que le pipeline gère correctement les valeurs manquantes sans planter."""
        X_with_nans = self.X_dummy.copy()
        X_with_nans.loc[0, "TotalCharges"] = np.nan
        X_with_nans.loc[1, "PaymentMethod"] = np.nan

        self.pipeline.fit(self.X_dummy, self.y_dummy)
        preds = self.pipeline.predict(X_with_nans)
        self.assertEqual(len(preds), len(X_with_nans))


if __name__ == "__main__":
    unittest.main()
