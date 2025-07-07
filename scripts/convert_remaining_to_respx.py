#!/usr/bin/env python3
"""Convert remaining test files to use respx for HTTP mocking."""

import re
from pathlib import Path


def add_respx_import(content: str) -> str:
    """Add respx import if not already present."""
    if 'import respx' not in content and 'from respx' not in content:
        # Find the last import line
        import_lines = []
        other_lines = []
        in_imports = True
        
        for line in content.split('\n'):
            if in_imports and (line.startswith('import ') or line.startswith('from ') or line.strip() == ''):
                import_lines.append(line)
            else:
                in_imports = False
                other_lines.append(line)
        
        # Add respx import
        import_lines.append('import respx')
        
        content = '\n'.join(import_lines + other_lines)
    
    return content


def convert_httpx_mocks_to_respx(content: str) -> str:
    """Convert httpx AsyncMock patterns to respx."""
    # Pattern: mock_httpx_client.get = AsyncMock(return_value=...)
    content = re.sub(
        r'mock_httpx_client\.get\s*=\s*AsyncMock\(return_value=Mock\(status_code=(\d+),\s*json=lambda:\s*({[^}]+})\)\)',
        r'respx_mock.get("").mock(return_value=httpx.Response(\1, json=\2))',
        content
    )
    
    # Pattern: mock_httpx_client.post = AsyncMock(return_value=...)
    content = re.sub(
        r'mock_httpx_client\.post\s*=\s*AsyncMock\(return_value=Mock\(status_code=(\d+),\s*json=lambda:\s*({[^}]+})\)\)',
        r'respx_mock.post("").mock(return_value=httpx.Response(\1, json=\2))',
        content
    )
    
    # Add @respx.mock decorator to test methods using httpx mocks
    lines = content.split('\n')
    new_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check if this is a test method that might use httpx
        if re.match(r'^\s*(async )?def test_', line):
            # Look ahead for httpx mock usage
            j = i + 1
            has_httpx_mock = False
            indent_level = len(line) - len(line.lstrip())
            
            while j < len(lines) and (lines[j].strip() == '' or len(lines[j]) - len(lines[j].lstrip()) > indent_level):
                if 'mock_httpx' in lines[j] or 'respx_mock' in lines[j]:
                    has_httpx_mock = True
                    break
                j += 1
            
            if has_httpx_mock:
                # Check if there's already a respx decorator
                k = i - 1
                has_respx_decorator = False
                while k >= 0 and (lines[k].strip().startswith('@') or lines[k].strip() == ''):
                    if '@respx.mock' in lines[k]:
                        has_respx_decorator = True
                        break
                    k -= 1
                
                if not has_respx_decorator:
                    # Add the decorator
                    indent = ' ' * indent_level
                    new_lines.append(f'{indent}@respx.mock')
        
        new_lines.append(line)
        i += 1
    
    return '\n'.join(new_lines)


def fix_test_health_checks(file_path: Path) -> None:
    """Fix HTTP mocking in test_health_checks.py."""
    content = file_path.read_text()
    
    if 'AsyncMock' in content and 'httpx' in content:
        content = add_respx_import(content)
        content = convert_httpx_mocks_to_respx(content)
        
        # Add httpx import if needed
        if 'import httpx' not in content:
            content = content.replace('import respx', 'import httpx\nimport respx')
        
        file_path.write_text(content)
        print(f"Converted {file_path} to use respx")
    else:
        print(f"No conversion needed for {file_path}")


def fix_test_files():
    """Fix test files to use respx."""
    base_path = Path("/workspace/repos/ai-docs-vector-db-hybrid-scraper")
    
    # List of test files that might need conversion
    test_files = [
        "tests/unit/utils/test_health_checks.py",
        "tests/unit/services/crawling/test_lightweight_scraper.py",
        "tests/unit/services/embeddings/test_crawl4ai_bulk_embedder.py",
        "tests/unit/mcp_tools/tools/test_search_utils.py",
        "tests/unit/mcp_tools/tools/test_documents.py",
    ]
    
    for file_path in test_files:
        full_path = base_path / file_path
        if full_path.exists():
            try:
                fix_test_health_checks(full_path)
            except Exception as e:
                print(f"Error converting {full_path}: {e}")
        else:
            print(f"File not found: {full_path}")


if __name__ == "__main__":
    fix_test_files()