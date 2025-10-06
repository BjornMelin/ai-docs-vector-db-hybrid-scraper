# Developer Setup Guide

## Prerequisites

### Python 3.11-3.12

#### Ubuntu/Debian

```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev
```

#### macOS with Homebrew

```bash
brew install python@3.11
```

#### Windows with Chocolatey

```bash
choco install python --version=3.11.9
```

### uv (package manager)

#### Linux/macOS

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Windows with PowerShell

```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Docker

#### Ubuntu/Debian

```bash
sudo apt install docker.io docker-compose
```

#### macOS with Homebrew

```bash
brew install docker docker-compose
```

#### Windows

Download Docker Desktop from https://www.docker.com/products/docker-desktop

### Git

#### Ubuntu/Debian

```bash
sudo apt install git
```

#### macOS with Homebrew

```bash
brew install git
```

#### Windows

Download Git from https://git-scm.com/download/win

## Repository Setup

### Clone repository

```bash
git clone https://github.com/BjornMelin/ai-docs-vector-db-hybrid-scraper.git
cd ai-docs-vector-db-hybrid-scraper
```

### Install dependencies

```bash
uv sync
```

### Copy environment file

```bash
cp .env.example .env
```

## Essential Environment Variables

### Edit .env with required keys

```bash
# Required variables only - add to .env file

OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
DATABASE_URL=postgresql://user:password@localhost:5432/aidocs
REDIS_URL=redis://localhost:6379/0
```

## Service Startup

### Start all services

```bash
docker-compose up -d
```

### Start specific services

```bash
docker-compose up -d postgres redis
```

### View service status

```bash
docker-compose ps
```

### View service logs

```bash
docker-compose logs -f
```

## Verification

### Run tests

```bash
uv run pytest
```

### Run specific test file

```bash
uv run pytest tests/test_scraper.py
```

### Run linting

```bash
uv run ruff check .
```

### Format code

```bash
uv run ruff format .
```

### Type checking

```bash
uv run mypy .
```

## Basic Development Workflow

### Create and switch to new branch

```bash
git checkout -b feature/new-feature
```

### Make changes and run tests

```bash
uv run pytest
```

### Commit changes

```bash
git add .
git commit -m "Add new feature"
```

### Push branch

```bash
git push origin feature/new-feature
```

### Run pre-commit checks before commit

```bash
uv run pre-commit run --all-files
```
