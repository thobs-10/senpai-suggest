#!/usr/bin/env python3
"""
Repository setup script for the Anime Recommender System project.
Creates the directory structure and root-level configuration files.
"""

import os
from pathlib import Path


def create_directories():
    """Create all necessary directories."""
    dirs = [
        ".github/workflows",
        "data",
        "pipelines/steps",
        "api/schemas",
        "ui",
        "infrastructure",
        "docker",
        "tests",
        "utilities",
        "src/senpai_suggest",
    ]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")


def create_gitkeep_files():
    """Add .gitkeep files to empty directories."""
    dirs_with_gitkeep = [
        "data",
        "pipelines/steps",
        "api/schemas",
        "ui",
        "infrastructure",
        "docker",
        "tests",
        "utilities",
        ".github/workflows",
    ]

    for dir_path in dirs_with_gitkeep:
        gitkeep_path = Path(dir_path) / ".gitkeep"
        gitkeep_path.touch()
        print(f"✓ Created .gitkeep: {dir_path}/.gitkeep")


def create_package_init_files():
    """Create __init__.py files for Python packages."""
    package_dirs = [
        "src/senpai_suggest",
    ]
    
    for package_dir in package_dirs:
        init_path = Path(package_dir) / "__init__.py"
        init_path.touch()
        print(f"✓ Created __init__.py: {package_dir}/__init__.py")


def create_root_files():
    """Create root-level configuration files."""

    # Makefile
    makefile_content = """.PHONY: help install test lint format clean run-local

help:
\t@echo "Available targets:"
\t@echo "  make install        - Install dependencies"
\t@echo "  make test           - Run tests"
\t@echo "  make lint           - Run linting checks"
\t@echo "  make format         - Format code with black and isort"
\t@echo "  make clean          - Clean up generated files"
\t@echo "  make run-local      - Run pipeline locally"

install:
\t@echo "Installing dependencies..."
\tuv pip install -e ".[dev]"

test:
\t@echo "Running tests..."
\tpytest tests/ -v --cov=pipelines --cov=api --cov=ui

lint:
\t@echo "Running linting..."
\tpylint pipelines api ui
\tflake8 pipelines api ui

format:
\t@echo "Formatting code..."
\tblack .
\tisort .

clean:
\t@echo "Cleaning up..."
\tfind . -type d -name __pycache__ -exec rm -rf {} +
\tfind . -type f -name "*.pyc" -delete
\trm -rf .pytest_cache .coverage htmlcov

run-local:
\t@echo "Running pipeline locally..."
\tcd pipelines && python run.py
"""

    with open("Makefile", "w") as f:
        f.write(makefile_content)
    print("✓ Created: Makefile")

    # pyproject.toml
    pyproject_content = """[build-system]
requires = ["setuptools>=65.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "senpai-suggest"
version = "0.1.0"
description = "End-to-end anime recommender system with MLOps pipeline"
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }
authors = [
    { name = "Author Name", email = "author@example.com" }
]
dependencies = [
    "zenml>=0.56.0",
    "comet-ml>=3.35.0",
    "implicit>=0.7.2",
    "scikit-learn>=1.3.0",
    "fastapi>=0.104.0",
    "uvicorn>=0.24.0",
    "streamlit>=1.28.0",
    "boto3>=1.28.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "black>=23.10.0",
    "isort>=5.12.0",
    "pylint>=3.0.0",
    "flake8>=6.1.0",
    "mypy>=1.6.0",
    "pre-commit>=3.5.0",
]

[tool.black]
line-length = 100
target-version = ["py310"]
include = '\\.pyi?$'
extend-exclude = '''
/(
  # directories
  \\.eggs
  | \\.git
  | \\.hg
  | \\.mypy_cache
  | \\.tox
  | \\.venv
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
line_length = 100
skip_gitignore = true
multi_line_mode = 3
include_trailing_comma = true

[tool.pylint]
max-line-length = 100

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false
check_untyped_defs = true
"""

    with open("pyproject.toml", "w") as f:
        f.write(pyproject_content)
    print("✓ Created: pyproject.toml")

    # .env.example
    env_example_content = """# Comet ML
COMET_API_KEY=<your_comet_api_key>
COMET_PROJECT_NAME=anime-recommender

# AWS Configuration
AWS_ACCESS_KEY_ID=<your_aws_access_key>
AWS_SECRET_ACCESS_KEY=<your_aws_secret_key>
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET_NAME=anime-recommender-bucket
MODEL_PATH=s3://anime-recommender-bucket/models/

# Environment
ENVIRONMENT=staging
DEBUG=false

# ZenML
ZENML_ANALYTICS_OPT_IN=false

# API Settings
API_HOST=0.0.0.0
API_PORT=8000

# Streamlit Settings
STREAMLIT_SERVER_HEADLESS=true
"""

    with open(".env.example", "w") as f:
        f.write(env_example_content)
    print("✓ Created: .env.example")

    # .gitignore
    gitignore_content = """# Virtual environments
venv/
env/
ENV/
.venv

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# Environment variables
.env
.env.local
.env.*.local

# Project-specific
data/raw/
data/processed/
models/
outputs/
logs/
*.mlflow

# ZenML
.zenml/
zenml_local_store/

# Comet ML
cometml.config

# AWS credentials
credentials/
*.pem

# Docker
*.log
docker-compose.override.yml

# Terraform
*.tfstate
*.tfstate.*
*.tfvars
.terraform/
.terraform.lock.hcl
"""

    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    print("✓ Created: .gitignore")

    # .pre-commit-config.yaml
    precommit_content = """repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict

  - repo: https://github.com/psf/black
    rev: 23.10.1
    hooks:
      - id: black
        language_version: python3.10

  - repo: https://github.com/PyCQA/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/PyCQA/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
        args: ["--max-line-length=100"]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.6.1
    hooks:
      - id: mypy
        additional_dependencies: ["types-all"]
        args: ["--ignore-missing-imports"]
"""

    with open(".pre-commit-config.yaml", "w") as f:
        f.write(precommit_content)
    print("✓ Created: .pre-commit-config.yaml")


def main():
    """Execute the setup."""
    print("🚀 Setting up Anime Recommender System repository structure...\n")

    try:
        create_directories()
        print()
        create_gitkeep_files()
        print()
        create_package_init_files()
        print()
        create_root_files()
        print("\n✅ Repository setup complete!")
        print("\nNext steps:")
        print("1. Review and update .env.example with your credentials")
        print("2. Run: make install")
        print("3. Run: pre-commit install")
    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
        raise


if __name__ == "__main__":
    main()
