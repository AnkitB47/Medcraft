.PHONY: setup dev dev_gpu test lint clean build push deploy eval_all

setup:
	pip install -r requirements.txt
	pre-commit install

dev:
	docker-compose up --build

dev_gpu:
	docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up --build

test:
	pytest tests/

lint:
	flake8 .
	black . --check

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:
	docker-compose build

push:
	docker-compose push

deploy:
	gh workflow run deploy.yml

eval_all:
	python eval/eval_yolo.py
	python eval/eval_nanovlm.py
	python eval/eval_titans.py
	python eval/generate_report.py
