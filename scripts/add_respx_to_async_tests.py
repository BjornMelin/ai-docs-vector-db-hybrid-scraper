#!/usr/bin/env python3
"""Add respx mocking to async HTTP tests."""

import ast
import re
from pathlib import Path
from typing import List, Set, Tuple

class AsyncHTTPTestFinder(ast.NodeVisitor):
    """Find async tests that make HTTP calls."""
    
    def __init__(self):
        self.async_http_tests = []
        self.current_class = None
        self.current_function = None
        self.imports = set()
        
    def visit_Import(self, node):
        """Track imports to identify HTTP libraries."""
        for alias in node.names:
            self.imports.add(alias.name)
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):
        """Track from imports."""
        if node.module:
            self.imports.add(node.module)
            for alias in node.names:
                self.imports.add(f"{node.module}.{alias.name}")
        self.generic_visit(node)
        
    def visit_ClassDef(self, node):
        """Track current class."""
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = None
        
    def visit_AsyncFunctionDef(self, node):
        """Find async test functions."""
        if node.name.startswith('test_'):
            self.current_function = node
            # Check if function uses HTTP
            if self._uses_http(node):
                test_info = {
                    'name': node.name,
                    'class': self.current_class,
                    'node': node,
                    'has_respx': self._has_respx_decorator(node),
                    'http_calls': self._find_http_calls(node)
                }
                self.async_http_tests.append(test_info)
        self.generic_visit(node)
        self.current_function = None
        
    def _uses_http(self, node):
        """Check if function makes HTTP calls."""
        # Look for common HTTP patterns
        http_patterns = [
            'httpx', 'aiohttp', 'requests', 'urllib',
            'ClientSession', 'AsyncClient', 'get(', 'post(',
            'put(', 'delete(', 'patch(', 'head('
        ]
        
        source = ast.unparse(node)
        return any(pattern in source for pattern in http_patterns)
        
    def _has_respx_decorator(self, node):
        """Check if function already uses respx."""
        for decorator in node.decorator_list:
            if 'respx' in ast.unparse(decorator):
                return True
        return False
        
    def _find_http_calls(self, node):
        """Find specific HTTP calls in the function."""
        calls = []
        
        class CallFinder(ast.NodeVisitor):
            def visit_Call(self, call_node):
                # Check for HTTP method calls
                call_str = ast.unparse(call_node)
                if any(method in call_str for method in 
                       ['get(', 'post(', 'put(', 'delete(', 'patch(']):
                    calls.append(call_str)
                self.generic_visit(call_node)
                
        CallFinder().visit(node)
        return calls


def add_respx_import(content: str) -> str:
    """Add respx import if not present."""
    lines = content.split('\n')
    
    # Check if respx is already imported
    if any('import respx' in line for line in lines):
        return content
        
    # Find the right place to add import
    import_added = False
    new_lines = []
    
    for i, line in enumerate(lines):
        if not import_added and (line.startswith('import ') or line.startswith('from ')):
            # Add after the last import
            if i + 1 < len(lines) and not (lines[i + 1].startswith('import ') or 
                                           lines[i + 1].startswith('from ')):
                new_lines.append(line)
                new_lines.append('import respx')
                import_added = True
                continue
        new_lines.append(line)
        
    if not import_added:
        # Add at the beginning after docstring
        for i, line in enumerate(new_lines):
            if not line.strip() or line.strip().startswith('"""'):
                continue
            new_lines.insert(i, 'import respx\n')
            break
            
    return '\n'.join(new_lines)


def add_respx_decorator(content: str, test_info: dict) -> str:
    """Add respx.mock decorator to async test."""
    if test_info['has_respx']:
        return content
        
    test_name = test_info['name']
    class_name = test_info['class']
    
    # Build the pattern to find the test
    if class_name:
        pattern = rf'(class {class_name}.*?:\n(?:.*\n)*?)(\s*)(async def {test_name}\b)'
    else:
        pattern = rf'(\n)(\s*)(async def {test_name}\b)'
        
    def add_decorator(match):
        prefix = match.group(1)
        indent = match.group(2)
        func_def = match.group(3)
        
        # Check if there's already a decorator
        lines_before = content[:match.start()].split('\n')
        if lines_before and '@' in lines_before[-1]:
            # Add respx decorator after existing decorators
            return f"{prefix}{indent}@respx.mock\n{indent}{func_def}"
        else:
            # Add as first decorator
            return f"{prefix}{indent}@respx.mock\n{indent}{func_def}"
            
    return re.sub(pattern, add_decorator, content, flags=re.MULTILINE | re.DOTALL)


def convert_http_mocks(content: str, test_info: dict) -> str:
    """Convert manual HTTP mocks to respx patterns."""
    # This is a simplified version - real conversion would need more context
    
    # Pattern 1: Convert patch('httpx.AsyncClient') to respx
    content = re.sub(
        r'@patch\(["\']httpx\.AsyncClient["\']\)',
        '@respx.mock',
        content
    )
    
    # Pattern 2: Convert mock response setup
    content = re.sub(
        r'mock_response\.json\.return_value = ({[^}]+})',
        r'respx.get("https://api.example.com").mock(return_value=httpx.Response(200, json=\1))',
        content
    )
    
    return content


def process_file(file_path: Path) -> Tuple[bool, List[str]]:
    """Process a single file for respx conversion."""
    try:
        content = file_path.read_text()
        tree = ast.parse(content)
        
        finder = AsyncHTTPTestFinder()
        finder.visit(tree)
        
        if not finder.async_http_tests:
            return False, []
            
        messages = []
        original_content = content
        
        # Add respx import
        content = add_respx_import(content)
        
        # Process each test
        for test_info in finder.async_http_tests:
            if not test_info['has_respx']:
                content = add_respx_decorator(content, test_info)
                messages.append(f"Added @respx.mock to {test_info['name']}")
                
                # Note: Real conversion would analyze the specific HTTP calls
                # and create appropriate respx mocks
                
        if content != original_content:
            file_path.write_text(content)
            return True, messages
            
        return False, []
        
    except Exception as e:
        return False, [f"Error: {e}"]


def main():
    """Main execution."""
    print("Adding respx mocking to async HTTP tests...")
    print()
    
    # Find all test files
    test_files = list(Path('tests').rglob('test_*.py'))
    
    processed = 0
    converted = 0
    
    for file_path in test_files:
        if file_path.is_file():
            processed += 1
            success, messages = process_file(file_path)
            
            if success:
                converted += 1
                print(f"✓ {file_path}")
                for msg in messages:
                    print(f"  - {msg}")
            elif messages:
                print(f"✗ {file_path}")
                for msg in messages:
                    print(f"  - {msg}")
                    
    print(f"\nProcessed {processed} files, converted {converted} files")
    print("\nNote: This script adds respx decorators but doesn't fully convert mock patterns.")
    print("Manual review is needed to complete the conversion of mock setups.")


if __name__ == "__main__":
    main()