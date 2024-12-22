.PHONY: install test lint format clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=thinking_llms --cov-report=term-missing

lint:
	flake8 thinking_llms tests
	black --check thinking_llms tests
	isort --check-only thinking_llms tests

format:
	black thinking_llms tests
	isort thinking_llms tests

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".hypothesis" -exec rm -rf {} + 