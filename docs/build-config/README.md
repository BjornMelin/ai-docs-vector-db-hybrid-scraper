# Documentation Build Configuration

This directory contains configuration files for building project documentation.

## Files

- **`mkdocs.yml`** - MkDocs Material theme configuration for user-friendly documentation
- **`conf.py`** - Sphinx configuration for API documentation generation
- **`index.rst`** - Sphinx index file for RST-based documentation

## Usage

### Building MkDocs Documentation

```bash
# From project root
uv run mkdocs build -f docs/build-config/mkdocs.yml
uv run mkdocs serve -f docs/build-config/mkdocs.yml  # Development server
```

### Building Sphinx API Documentation

```bash
# From project root
uv run sphinx-apidoc -o docs/api src --force
uv run sphinx-build -b html -c docs/build-config docs docs/_build/html
```

## Automated Builds

The GitHub Actions workflow in `.github/workflows/docs.yml` automatically builds
both documentation formats and deploys them to GitHub Pages.
