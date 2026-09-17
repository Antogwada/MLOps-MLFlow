# Dockerfile pour le projet Churn Classifier avec MLflow
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Installation de curl et dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# Exposer le port de l'UI MLflow
EXPOSE 5000

# Commande par défaut : exécuter l'évaluation ou lancer le serveur MLflow
CMD ["mlflow", "ui", "--host", "0.0.0.0", "--port", "5000"]
