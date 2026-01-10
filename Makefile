.PHONY: help install install-dev test test-cov lint format clean build docs docker

# =============================================================================
# FaceVerify Makefile
# =============================================================================
# Production-ready commands for development, testing, and deployment
# =============================================================================

# Default target
help:
	@echo "FaceVerify - Available Commands"
	@echo "================================"
	@echo ""
	@echo "Installation:"
	@echo "  install          Install package in production mode"
	@echo "  install-dev      Install with development dependencies"
	@echo "  install-gpu      Install with GPU support"
	@echo "  uninstall        Uninstall the package"
	@echo ""
	@echo "Testing:"
	@echo "  test             Run all tests"
	@echo "  test-unit        Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-cov         Run tests with coverage report"
	@echo "  test-fast        Run tests excluding slow tests"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint             Run all linters (flake8, mypy)"
	@echo "  format           Format code with black and isort"
	@echo "  format-check     Check code formatting without changes"
	@echo "  check            Run format-check and lint"
	@echo ""
	@echo "Build:"
	@echo "  build            Build source and wheel distribution"
	@echo "  build-wheel      Build wheel only"
	@echo "  docs             Build documentation"
	@echo ""
	@echo "Docker:"
	@echo "  docker           Build Docker image"
	@echo "  docker-run       Run Docker container"
	@echo "  docker-up        Start with docker-compose"
	@echo "  docker-down      Stop docker-compose services"
	@echo "  docker-gpu       Start with GPU support"
	@echo ""
	@echo "Utility:"
	@echo "  clean            Clean all build artifacts"
	@echo "  clean-pyc        Clean Python cache files"
	@echo "  clean-test       Clean test artifacts"
	@echo "  verify           Run example verification"
	@echo "  api              Start REST API server"
	@echo "  notebook         Start Jupyter notebook"

# =============================================================================
# Installation
# =============================================================================

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install || true

install-gpu:
	pip install -e ".[gpu]"

install-all:
	pip install -e ".[all]"

uninstall:
	pip uninstall faceverify -y

reinstall: uninstall install

# =============================================================================
# Testing
# =============================================================================

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v -m integration

test-cov:
	pytest tests/ -v --cov=src/faceverify --cov-report=html --cov-report=term-missing

test-fast:
	pytest tests/ -v -m "not slow"

test-gpu:
	pytest tests/ -v -m gpu

# =============================================================================
# Code Quality
# =============================================================================

lint:
	flake8 src/faceverify tests/ || true
	mypy src/faceverify || true

format:
	black src/faceverify tests/ examples/
	isort src/faceverify tests/ examples/

format-check:
	black --check src/faceverify tests/ examples/
	isort --check-only src/faceverify tests/ examples/

check: format-check lint

# Pre-commit hook
pre-commit:
	pre-commit run --all-files

# =============================================================================
# Build
# =============================================================================

build: clean
	python -m build

build-wheel: clean
	python -m build --wheel

build-sdist: clean
	python -m build --sdist

docs:
	cd docs && make html || echo "Sphinx not configured"

# =============================================================================
# Docker
# =============================================================================

docker:
	docker build -t faceverify:latest .

docker-run:
	docker run -p 8000:8000 --name faceverify-api faceverify:latest

docker-stop:
	docker stop faceverify-api || true
	docker rm faceverify-api || true

docker-up:
	docker-compose up -d faceverify

docker-down:
	docker-compose down

docker-gpu:
	docker-compose --profile gpu up -d faceverify-gpu

docker-logs:
	docker-compose logs -f

docker-clean:
	docker-compose down --rmi local -v

# =============================================================================
# Cleanup
# =============================================================================

clean: clean-build clean-pyc clean-test

clean-build:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf src/*.egg-info/

clean-pyc:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name "*~" -delete 2>/dev/null || true

clean-test:
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml

# =============================================================================
# Utility
# =============================================================================

verify:
	python examples/basic_verification.py

api:
	uvicorn examples.rest_api_server:app --reload --host 0.0.0.0 --port 8000

notebook:
	jupyter notebook notebooks/

# Version info
version:
	python -m faceverify --version

info:
	python -m faceverify info

# =============================================================================
# Release (for maintainers)
# =============================================================================

release-test: clean build
	twine upload --repository testpypi dist/*

release: clean build
	twine upload dist/*
