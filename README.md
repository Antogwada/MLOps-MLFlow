# 📊 Churn Classifier with Pipelines + MLflow

Projet MLOps complet conforme aux spécifications de `proposal.md` utilisant le jeu de données **Telco Customer Churn**.

Le projet combine :
* **Pipeline scikit-learn reproductible** (`ColumnTransformer` avec gestion automatique des variables numériques et catégorielles).
* **Validation croisée et recherche par grille** (`GridSearchCV` avec optimisation du `ROC-AUC`).
* **Suivi d'expérimentation avec MLflow** (`mlflow.autolog`, logging des hyperparamètres, métriques et modèles).
* **Enregistrement et traçabilité des artefacts** (Courbes ROC, Precision-Recall, Matrice de confusion, prédictions d'erreurs).
* **Registre de modèles MLflow** (`Model Registry` sous le nom `ChurnClassifier`).
* **Makefile complet**, tests unitaires (`pytest`), et conteneurisation `Dockerfile`.

---

## 📁 Structure du Répertoire

```text
MLOps-MLFlow/
├─ data/
│  ├─ raw.csv                    # Dataset brut Telco Customer Churn (gitignored)
│  └─ processed/                 # Splits train.csv et test.csv
├─ configs/
│  └─ config.yaml                # Configuration (chemins, features, hyperparamètres, MLflow)
├─ src/
│  ├─ config.py                  # Classe typée de configuration
│  ├─ get_data.py                # Téléchargement automatique du dataset
│  ├─ preprocess.py              # Nettoyage et encodage des données
│  ├─ pipeline.py                # Pipeline ColumnTransformer + Modèle
│  ├─ train.py                   # Entraînement avec GridSearchCV + MLflow autolog + registry
│  ├─ evaluate.py                # Évaluation sur le test set et logging des artefacts graphiques
│  ├─ utils.py                   # Génération des courbes ROC, PR et matrice de confusion
│  └─ predict.py                 # Script d'inférence par lot (batch)
├─ tests/
│  └─ test_pipeline.py           # Tests unitaires et sanity checks du pipeline
├─ artifacts/                    # Artefacts sauvegardés (graphiques PNG et prédictions)
├─ mlruns/                       # Stockage local des expériences MLflow
├─ Makefile                      # Automatisation complète du workflow
├─ requirements.txt              # Dépendances Python
├─ Dockerfile                    # Image de conteneurisation Docker
└─ README.md                     # Documentation complète
```

---

## 🚀 Utilisation Rapide via le `Makefile`

### 1. Installation et Environnement
```bash
make init
make install
```

### 2. Exécution du Pipeline Complet en 1 commande
```bash
make all
```
*Cette commande télécharge le dataset, prétraite les données, entraîne le modèle avec `GridSearchCV` et autologging MLflow, évalue les performances sur le test set et lance les tests unitaires.*

---

## 🔬 Étapes Détaillées

### 1. Télécharger les données
```bash
make get_data
```
Télécharge le dataset Telco Churn (7 043 lignes, 21 colonnes) dans `data/raw.csv`.

### 2. Prétraiter les données
```bash
make preprocess
```
Nettoie les valeurs manquantes dans `TotalCharges`, encode la cible binaire `Churn` et sépare en `train.csv` (80%) et `test.csv` (20%) de façon stratifiée.

### 3. Entraîner et Tracker avec MLflow
```bash
make train
```
Exécute la recherche par grille avec `GridSearchCV` (5 folds stratifiés), optimise le ROC-AUC, enregistre tous les runs dans MLflow et enregistre le meilleur modèle dans le Model Registry (`ChurnClassifier`).

### 4. Évaluer et Enregistrer les Artefacts
```bash
make evaluate
```
Calcule les métriques finales sur `data/processed/test.csv`, génère les graphiques d'analyse dans `artifacts/` (`roc_curve.png`, `pr_curve.png`, `confusion_matrix.png`, `test_predictions.csv`) et les logge dans MLflow.

### 5. Lancer l'Interface Graphique MLflow
```bash
make mlflow_ui
```
Ouvre l'interface de visualisation MLflow sur **[http://localhost:5000](http://localhost:5000)** pour comparer les runs, visualiser les courbes et inspecter le registre de modèles.

### 6. Tests Unitaires
```bash
make test
```

### 7. Inférence par Lot (Batch Inference)
Pour prédire le churn sur de nouveaux clients :
```bash
python src/predict.py --input data/processed/test.csv --output predictions.csv
```

---

## 🐳 Conteneurisation Docker

Construire l'image Docker :
```bash
make docker_build
```

Lancer le conteneur :
```bash
make docker_run
```
L'UI MLflow sera accessible sur [http://localhost:5000](http://localhost:5000).
