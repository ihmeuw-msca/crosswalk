# Makefile for simple installation and testing
.PHONY: build install install-dev tests clean uninstall

# Build the source distribution and wheel
build: pyproject.toml
	python -m build

# Standard installation
install: pyproject.toml
	python -m pip install .

# Editable install with testing dependencies for local development
install-dev: pyproject.toml
	python -m pip install -e ".[test]"

# Run tests
tests:
	python -m pytest tests

# Clean up build artifacts and compiled files
clean:
	find . -type f -name "*.so*" -delete
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf build/ dist/ MANIFEST

# Safely uninstall the package
uninstall:
	python -m pip uninstall -y crosswalk