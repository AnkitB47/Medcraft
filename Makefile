# MedCraft Makefile

.PHONY: dev dev_gpu eval_all deploy_azure test lint clean help

help:
	@echo "MedCraft Platform Makefile"
	@echo "Usage:"
	@echo "  make dev          - Run local CPU stack"
	@echo "  make dev_gpu      - Run local GPU stack"
	@echo "  make eval_all     - Run all evaluations and generate report"
	@echo "  make deploy_azure - Provision and deploy to Azure"
	@echo "  make test         - Run all tests"
	@echo "  make lint         - Run linting"
	@echo "  make clean        - Clean up temporary files"

dev:
	docker-compose up --build

dev_gpu:
	docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up --build

eval_all:
	@echo "Running evaluation pipeline..."
	# Trigger Airflow DAG or run local script
	python eval/run_all.py

deploy_azure:
	@echo "Deploying to Azure..."
	cd ops/terraform/azure && terraform init && terraform apply -auto-approve
	# Helm deploy logic here

test:
	pytest services/api services/vision_yolo services/nanovlm services/titans_memory

lint:
	flake8 .
	black --check .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf .coverage
