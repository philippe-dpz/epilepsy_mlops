#!/bin/bash
set -e

docker compose pull

echo "Démarrage de Prefect Server et MLflow..."
docker compose up -d prefect_server mlflow

echo "Attente de Prefect Server sur le port 4200..."
until curl -sSf http://localhost:4200/api > /dev/null; do
  echo "En attente de Prefect Server..."
  sleep 5
done

echo "Attente de MLflow sur le port 5000..."
until curl -sSf http://localhost:5000 > /dev/null; do
  echo "En attente de MLflow..."
  sleep 5
done

echo "Démarrage des autres services..."
docker compose up -d prefect_orchestrator auth_api inference_api patient_data_pull prometheus grafana

echo "Tous les services sont démarrés !"

echo "URLs disponibles:"
echo "- Prefect UI: http://localhost:4200"
echo "- MLflow UI: http://localhost:5000" 
echo "- Auth API: http://localhost:8000"
echo "- Inference API: http://localhost:8001"
echo "- Prometheus: http://localhost:9090"
echo "- Grafana: http://localhost:3002"
