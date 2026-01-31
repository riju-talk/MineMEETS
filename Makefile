.PHONY: help install install-dev test lint format check run docker-build docker-run docker-stop clean

# Default target
help:
	@echo "MineMEETS - MLOps Meeting Intelligence Platform"
	@echo ""
	@echo "Available targets:"
	@echo "  install       - Install production dependencies"
	@echo "  install-dev   - Install development dependencies"
	@echo "  test          - Run tests with coverage"
	@echo "  lint          - Run linting checks (pylint)"
	@echo "  format        - Format code with black"
	@echo "  check         - Run all quality checks (format, lint, test)"
	@echo "  run           - Run application locally"
	@echo "  docker-build  - Build Docker image"
	@echo "  docker-run    - Run application in Docker"
	@echo "  docker-stop   - Stop Docker containers"
	@echo "  clean         - Clean up generated files"

# Install production dependencies
install:
	pip install -r requirements.txt

# Install development dependencies
install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"

# Run tests with coverage
test:
	pytest tests/ -v --cov=agents --cov-report=term-missing --cov-report=html

# Run linting
lint:
	pylint agents/ app.py --max-line-length=100 --disable=C0114,C0115,C0116

# Format code
format:
	black agents/ app.py tests/ --line-length=100

# Type checking
typecheck:
	mypy agents/ app.py --ignore-missing-imports

# Run all quality checks
check: format lint test
	@echo "✅ All quality checks passed!"

# Run application locally
run:
	python app.py

# Build Docker image
docker-build:
	docker build -t minemeets:latest .

# Run application in Docker
docker-run:
	docker-compose up --build

# Stop Docker containers
docker-stop:
	docker-compose down

# Clean up generated files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf .mypy_cache
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info
	@echo "✅ Cleaned up generated files"

# Database operations (be careful with these!)
db-stats:
	python -c "from agents.pinecone_db import PineconeDB; db = PineconeDB(); print(db.get_index_stats())"

db-flush:
	@echo "⚠️  WARNING: This will delete ALL data from the database!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		python -c "from agents.pinecone_db import PineconeDB; db = PineconeDB(); db.flush_database()"; \
		echo "\n✅ Database flushed"; \
	else \
		echo "\n❌ Cancelled"; \
	fi
