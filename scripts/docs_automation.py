#!/usr/bin/env python3
"""Documentation automation script for AI Docs Vector DB."""

import subprocess
import sys
from pathlib import Path

import click


def update_status_indicators():
    """Update project status indicators automatically."""
    click.echo("📝 Updating status indicators...")
    result = subprocess.run([
        "python", "scripts/add_status_indicators.py"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        click.echo("  ✅ Status indicators updated")
    else:
        click.echo(f"  ❌ Failed to update status indicators: {result.stderr}")
        return False
    return True


def update_documentation_links():
    """Update all documentation links automatically."""
    click.echo("🔗 Updating documentation links...")
    result = subprocess.run([
        "python", "scripts/update_doc_links.py"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        click.echo("  ✅ Documentation links updated")
    else:
        click.echo(f"  ❌ Failed to update documentation links: {result.stderr}")
        return False
    return True


def validate_documentation_links():
    """Validate all documentation links."""
    click.echo("🔍 Validating documentation links...")
    result = subprocess.run([
        "python", "scripts/validate_docs_links.py"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        click.echo("  ✅ Documentation links validated")
    else:
        click.echo(f"  ❌ Documentation link validation failed: {result.stderr}")
        return False
    return True


def build_documentation():
    """Build documentation with MkDocs."""
    click.echo("📚 Building documentation...")
    result = subprocess.run([
        "mkdocs", "build"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        click.echo("  ✅ Documentation built successfully")
    else:
        click.echo(f"  ❌ Documentation build failed: {result.stderr}")
        return False
    return True


@click.command()
@click.option('--skip-validation', is_flag=True, help="Skip link validation")
@click.option('--build-only', is_flag=True, help="Only build, skip updates")
def main(skip_validation: bool, build_only: bool):
    """Complete documentation build process."""
    click.echo("📚 Starting documentation automation...\n")
    
    success = True
    
    if not build_only:
        # Update status indicators
        if not update_status_indicators():
            success = False
            
        # Update documentation links
        if not update_documentation_links():
            success = False
    
    # Validate links (unless skipped)
    if not skip_validation:
        if not validate_documentation_links():
            success = False
    
    # Build documentation
    if not build_documentation():
        success = False
    
    click.echo()
    if success:
        click.echo("✅ Documentation automation completed successfully!")
    else:
        click.echo("❌ Documentation automation failed!")
        
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()