# ==============================================================================
# Makefile - Churn Classifier with Pipelines + MLflow
# ==============================================================================

ifeq ($(OS),Windows_NT)
    VENV_BIN = venv/Scripts
else
    VENV_BIN = venv/bin
endif

SYSTEM_PYTHON = python
VENV_PYTHON = $(VENV_BIN)/python
VENV_PIP = $(VENV_BIN)/pip

CONFIG = configs/config.yaml
EXP ?= telco-churn-classifier
PORT ?= 5000

.PHONY: help init install get_data preprocess train evaluate test mlflow_ui docker_build docker_run clean all

help:
	@echo "Commandes disponibles :"
	@echo "  make init          - Crée l'environnement virtuel venv"
	@echo "  make install       - Installe les dépendances Python requises"
	@echo "  make get_data      - Télécharge le dataset Telco Customer Churn"
	@echo "  make preprocess    - Prétraite les données et sépare en train/test"
	@echo "  make train         - Entraîne avec GridSearchCV et autolog MLflow"
	@echo "  make evaluate      - Évalue le modèle sur le jeu de test et log les artefacts"
	@echo "  make test          - Exécute les tests unitaires pytest"
	@echo "  make mlflow_ui     - Lance l'interface graphique de tracking MLflow"
	@echo "  make docker_build  - Construit l'image Docker du projet"
	@echo "  make docker_run    - Exécute le conteneur Docker"
	@echo "  make clean         - Nettoie les caches et fichiers temporaires"
	@echo "  make all           - Exécute l'intégralité du pipeline (data, train, eval, test)"

init:
	$(SYSTEM_PYTHON) -m venv venv
	@echo "Environnement virtuel créé dans ./venv"
	@echo "Activez-le via : source $(VENV_BIN)/activate (ou .\\venv\\Scripts\\Activate.ps1)"

install:
	$(VENV_PIP) install -r requirements.txt

get_data:
	$(VENV_PYTHON) src/get_data.py --config $(CONFIG)

preprocess: get_data
	$(VENV_PYTHON) src/preprocess.py --config $(CONFIG)

train: preprocess
	$(VENV_PYTHON) src/train.py --config $(CONFIG)

evaluate: train
	$(VENV_PYTHON) src/evaluate.py --config $(CONFIG)

test:
	$(VENV_PYTHON) -m pytest -v tests/

mlflow_ui:
	$(VENV_PYTHON) -m mlflow ui --port $(PORT)

docker_build:
	docker build -t churn-mlflow:latest .

docker_run:
	docker run -p 5000:5000 churn-mlflow:latest

clean:
	rm -rf data/processed/*
	rm -rf artifacts/*
	rm -rf __pycache__ src/__pycache__ tests/__pycache__ .pytest_cache

all: get_data preprocess train evaluate test
