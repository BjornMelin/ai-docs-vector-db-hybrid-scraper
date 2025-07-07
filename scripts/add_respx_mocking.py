#!/usr/bin/env python3
"""Script to add respx for HTTP mocking in test files."""

import re
from pathlib import Path
from typing import List, Tuple


def add_respx_import(content: str) -> Tuple[str, bool]:
    """Add respx import if not present."""
    if "import respx" in content or "from respx import" in content:
        return content, False
    
    lines = content.split('\n')
    import_added = False
    
    # Find where to add the import
    for i, line in enumerate(lines):
        if line.startswith('import pytest') or line.startswith('from unittest'):
            # Add respx import after pytest or unittest imports
            lines.insert(i + 1, "import respx")
            import_added = True
            break
    
    if not import_added:
        # Add at the beginning after docstring
        for i, line in enumerate(lines):
            if not line.startswith('"""') and not line.startswith('#') and line.strip():
                lines.insert(i, "import respx\n")
                import_added = True
                break
    
    return '\n'.join(lines), import_added


def convert_manual_mocks_to_respx(content: str) -> Tuple[str, List[str]]:
    """Convert manual HTTP mocks to respx patterns."""
    changes = []
    lines = content.split('\n')
    modified_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Pattern: mock_client.get = AsyncMock(return_value=...)
        match = re.match(r'^(\s*)mock_(\w+)\.(\w+)\s*=\s*AsyncMock\(return_value=(.*)\)', line)
        if match:
            indent = match.group(1)
            method = match.group(3).lower()
            return_value = match.group(4)
            
            # Add respx decorator or context manager if not already present
            # Look for the test function definition above
            func_line = None
            for j in range(i-1, max(0, i-20), -1):
                if re.match(r'^(\s*)(async\s+)?def\s+test_', lines[j]):
                    func_line = j
                    break
            
            if func_line is not None:
                # Check if already has respx decorator
                has_respx = False
                for j in range(max(0, func_line-5), func_line):
                    if '@respx.mock' in lines[j]:
                        has_respx = True
                        break
                
                if not has_respx:
                    # Add respx decorator
                    func_indent = re.match(r'^(\s*)', lines[func_line]).group(1)
                    lines[func_line] = f"{func_indent}@respx.mock\n{lines[func_line]}"
                    changes.append("Added @respx.mock decorator")
                    i += 1  # Account for the added line
            
            # Convert to respx route
            if method in ['get', 'post', 'put', 'patch', 'delete']:
                # Create respx route
                modified_lines.append(f"{indent}respx.{method}('https://api.example.com/').mock(return_value=httpx.Response(200, json={return_value}))")
                changes.append(f"Converted manual {method.upper()} mock to respx")
            else:
                # Keep original line if not a standard HTTP method
                modified_lines.append(line)
            
            i += 1
            continue
        
        # Pattern: mock_response = AsyncMock()
        if 'mock_response' in line and 'AsyncMock()' in line:
            # Skip this line as we'll handle it with respx
            changes.append("Removed manual mock_response creation")
            i += 1
            continue
        
        modified_lines.append(line)
        i += 1
    
    # Add httpx import if using respx
    if changes and 'import httpx' not in '\n'.join(modified_lines):
        # Find where to add httpx import
        for i, line in enumerate(modified_lines):
            if line.strip() == 'import respx':
                modified_lines.insert(i + 1, "import httpx")
                break
    
    return '\n'.join(modified_lines), changes


def process_file(file_path: Path) -> dict:
    """Process a single file to add respx mocking."""
    try:
        content = file_path.read_text()
        original_content = content
        
        changes = []
        
        # Add respx import
        content, import_added = add_respx_import(content)
        if import_added:
            changes.append("Added respx import")
        
        # Convert manual mocks to respx
        content, mock_changes = convert_manual_mocks_to_respx(content)
        changes.extend(mock_changes)
        
        # Write back if changes were made
        if content != original_content:
            file_path.write_text(content)
            return {
                'file': str(file_path),
                'changes': changes,
                'success': True
            }
        else:
            return {
                'file': str(file_path),
                'changes': [],
                'success': True
            }
            
    except Exception as e:
        return {
            'file': str(file_path),
            'error': str(e),
            'success': False
        }


def main():
    """Main entry point."""
    # Files identified as needing respx
    files_to_fix = [
        "/workspace/repos/ai-docs-vector-db-hybrid-scraper/tests/unit/services/cache/test_browser_cache.py",
        "/workspace/repos/ai-docs-vector-db-hybrid-scraper/tests/unit/services/embeddings/test_crawl4ai_bulk_embedder.py",
        "/workspace/repos/ai-docs-vector-db-hybrid-scraper/tests/unit/services/embeddings/test_crawl4ai_bulk_embedder_extended.py",
        "/workspace/repos/ai-docs-vector-db-hybrid-scraper/tests/unit/mcp_tools/tools/test_search_utils.py",
        "/workspace/repos/ai-docs-vector-db-hybrid-scraper/tests/unit/mcp_tools/tools/test_documents.py",
    ]
    
    print("🔧 Adding respx for HTTP mocking in test files...\n")
    
    total_fixed = 0
    for file_path in files_to_fix:
        path = Path(file_path)
        if not path.exists():
            print(f"❌ File not found: {file_path}")
            continue
            
        result = process_file(path)
        
        if result.get('error'):
            print(f"❌ Error processing {result['file']}: {result['error']}")
        elif result['changes']:
            total_fixed += 1
            print(f"✅ Updated {result['file']}:")
            for change in result['changes']:
                print(f"   - {change}")
        else:
            print(f"ℹ️  No changes needed in {result['file']}")
        print()
    
    print(f"\n📊 Summary: Updated {total_fixed} files")


if __name__ == "__main__":
    main()