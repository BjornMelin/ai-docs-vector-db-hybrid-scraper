# Unified Configuration Guide

## Configuration Layout

config/
├── base.py              # Shared configuration
├── simple.py           # Development settings
├── enterprise.py       # Production settings
├── test.py             # Testing settings
├── secrets.py          # Secret management
└── __init__.py

## Essential Environment Variables

| Variable | Purpose | Example |
|---------|---------|---------|
| API_HOST | API server host | localhost |
| API_PORT | API server port | 8000 |
| QDRANT_URL | Vector database URL | http://localhost:6333 |
| DATABASE_URL | Primary database | postgresql://user:pass@host:5432/db |
| REDIS_URL | Cache/queue backend | redis://localhost:6379 |
| LOG_LEVEL | Application logging | INFO |
| SECRET_KEY | Cryptographic key | your-secret-key-here |
| DEBUG | Debug mode toggle | False |

## Environment Profiles

### Simple (Development)
- Single node setup
- SQLite database
- Debug enabled
- Local Qdrant instance
- Console logging

### Enterprise (Production)
- Multi-node deployment
- PostgreSQL database
- Redis caching
- External Qdrant service
- Structured logging

### Test
- In-memory database
- Mock external services
- Verbose logging
- Test-specific ports

## Core Tool Configurations

[tool.ruff]
line-length = 88
select = ["E", "F", "I"]
ignore = ["E501"]
target-version = "py39"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --tb=short"

[tool.uv]
dev-dependencies = [
    "pytest>=7.0",
    "ruff>=0.1",
    "black>=23.0"
]

## Database Connection Settings

PostgreSQL:
postgresql://username:password@host:port/database

SQLite (dev):
sqlite:///./local.db

Redis (cache):
redis://localhost:6379/0

## Secrets Management

1. Use environment variables for all secrets
2. Never commit secrets to version control
3. Use secret management tools (Vault, AWS Secrets Manager)
4. Rotate secrets regularly
5. Encrypt secrets at rest

Example secrets.py:
import os
SECRET_KEY = os.environ.get("SECRET_KEY")
DATABASE_PASSWORD = os.environ.get("DB_PASSWORD")

## Configuration Overrides

Override via environment variables:
export API_HOST=production-server
export DATABASE_URL=postgresql://prod-user:pass@db-host:5432/prod-db

Override in profile files:
# enterprise.py
from .base import *
DEBUG = False
ALLOWED_HOSTS = ["production-domain.com"]

Runtime override example:
# In application code
config.API_HOST = os.getenv("API_HOST", config.API_HOST)