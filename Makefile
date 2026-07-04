.PHONY: help install test lint format clean run-local docs

help:
	@echo "Available targets:"
	@echo "  make install        - Install dependencies"
	@echo "  make test           - Run tests"
	@echo "  make lint           - Run linting checks"
	@echo "  make format         - Format code with ruff"
	@echo "  make clean          - Clean up generated files"
	@echo "  make run-local      - Run pipeline locally"
	@echo "  make docs           - Generate documentation with pdoc"

install:
	@echo "Installing dependencies..."
	uv pip install -e ".[dev]"

test:
	@echo "Running tests..."
	pytest tests/ -v --cov=pipelines --cov=api --cov=ui

lint:
	@echo "Running linting..."
	pylint pipelines api ui
	flake8 pipelines api ui

format:
	@echo "Formatting code..."
	ruff format .
	ruff check . --fix

clean:
	@echo "Cleaning up..."
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov

run-local:
	@echo "Running pipeline locally..."
	cd pipelines && python run.py

docs:
	@echo "Generating documentation..."
	pdoc -d google src/senpai_suggest --output-directory docs
	@echo "✓ Documentation generated in docs/ directory"
