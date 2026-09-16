
#  “Churn Classifier with Pipelines + MLflow”


Build a robust tabular classification pipeline (scikit-learn `Pipeline` + `ColumnTransformer`) and **track everything with MLflow** (params, metrics, artifacts, and model registry).
Choose one of these small, public, easy-to-use datasets (CSV):

* **Telco Customer Churn** (binary classification, mixed categorical/numeric)
* **Bank Marketing** (deposit yes/no)
* **Adult Income** (>=50K vs <50K)

> Target skills: reproducible training with Pipelines, feature preprocessing, cross-validation & tuning, MLflow tracking/registry, CLI + config, Makefile, and (optional) Docker.

---

## What you’ll deliver

**Repo layout**

```
mlops-project/
├─ data/                         # raw & processed (gitignored)
├─ configs/
│  └─ config.yaml                # paths, target, split, model/type/params
├─ src/
│  ├─ config.py                # get the configuration as class
│  ├─ get-data.py                # download the data
│  ├─ preprocess.py              # preprocess the data
│  ├─ pipeline.py                # build ColumnTransformer + model
│  ├─ train.py                   # train + tune + MLflow autolog + registry
│  ├─ evaluate.py                # final eval + plot & log artifacts
│  └─ utils.py                   # small helpers (split, plotting, etc.)
├─ tests/
│  └─ test_pipeline.py           # sanity checks for pipeline
├─ Makefile
├─ requirements.txt (or pyproject.toml)
├─ .env.example                  # MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
├─ Dockerfile
└─ README.md
```


**Makefile (example)**

```make
PY=python
ENV?=.venv
EXP?=churn-exp

init:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip -r requirements.txt


train:
	MLFLOW_EXPERIMENT_NAME=$(EXP) $(PY) src/train.py --config configs/config.yaml

evaluate:
	MLFLOW_EXPERIMENT_NAME=$(EXP) $(PY) src/evaluate.py --config configs/config.yaml


test:
	pytest -q
```

**configs/config.yaml (example)**

```yaml
data:
  csv_path: data/raw.csv
  target: churn
  test_size: 0.2
  random_state: 42

features:
  numeric: ["tenure", "MonthlyCharges", "TotalCharges"]
  categorical: ["gender", "SeniorCitizen", "Partner", "Dependents",
                "PhoneService", "InternetService", "Contract",
                "PaperlessBilling", "PaymentMethod"]

model:
  type: "logreg"  # or "random_forest"
  params:
    C: [0.1, 1.0, 10.0]
    penalty: ["l2"]
    solver: ["lbfgs", "liblinear"]

cv:
  strategy: "StratifiedKFold"
  n_splits: 5
  scoring: "roc_auc"
```

---

---

## Tasks

1. **Setup & Data**

   * Create repo, venv, install deps (`scikit-learn`, `pandas`, `mlflow`, `matplotlib`, `pyyaml`, `ruff`, `pytest`).
   * Place CSV in `data/raw.csv`.

2. **Baseline Pipeline**

   * Implement `pipeline.py`.
   * Quick train/test split; single model; baseline ROC-AUC; verify pipeline works.

3. **MLflow Integration + Tuning**

   * Enable `mlflow.autolog`.
   * Add `GridSearchCV` with proper param grid; log best metrics; log best model; check MLflow UI.

4. **Artifacts & Evaluation**

   * Compute & log ROC/PR plots, confusion matrix as artifacts.
   * Save predictions CSV for error analysis (optional).


6. **(Optional) Registry & Docker**

   * Register model to `ChurnClassifier` and set stage “Staging”.
   * Dockerfile to run training or a tiny `predict.py` script for batch inference.