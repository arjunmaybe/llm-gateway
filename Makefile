.PHONY: install test lint typecheck run docker-build

PY ?= py -V:3.11 -m

install:
	$(PY) pip install -r requirements.txt

test:
	$(PY) pytest -q

lint:
	$(PY) ruff check src tests

typecheck:
	$(PY) mypy --strict src

run:
	$(PY) uvicorn src.main:app --host 127.0.0.1 --port 8000

docker-build:
	docker build -t llm-gateway:0.1.0 .
