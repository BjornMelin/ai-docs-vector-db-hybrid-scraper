#!/usr/bin/env python3
"""Modernize async testing patterns across the codebase.

This script identifies and fixes async test anti-patterns:
1. Replaces patch/mock of httpx with respx
2. Ensures proper async/await usage
3. Adds async context managers
4. Implements proper async cleanup
5. Creates reusable async test utilities
"""

import ast
import re
from pathlib import Path
from typing import List, Tuple, Dict, Set
import click
import subprocess


class AsyncTestModernizer(ast.NodeVisitor):
    """AST visitor to identify async test patterns that need modernization."""
    
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.issues = []
        self.has_respx_import = False
        self.has_httpx_import = False
        self.async_functions = set()
        self.patch_targets = []
        self.missing_async_await = []
        self.improper_cleanup = []
        
    def visit_Import(self, node: ast.Import) -> None:
        """Check for imports."""
        for alias in node.names:
            if alias.name == 'respx':
                self.has_respx_import = True
            elif alias.name == 'httpx':
                self.has_httpx_import = True
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Check for from imports."""
        if node.module == 'respx':
            self.has_respx_import = True
        elif node.module == 'httpx':
            self.has_httpx_import = True
        self.generic_visit(node)
        
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Check function definitions."""
        if node.name.startswith('test_'):
            self.check_async_test_patterns(node)
        self.generic_visit(node)
        
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Check async function definitions."""
        if node.name.startswith('test_'):
            self.async_functions.add(node.name)
            self.check_async_test_patterns(node)
        self.generic_visit(node)
        
    def check_async_test_patterns(self, node):
        """Check for various async test patterns."""
        # Check for patch/mock of httpx
        for decorator in node.decorator_list:
            if self.is_patch_decorator(decorator):
                target = self.get_patch_target(decorator)
                if target and 'httpx' in target:
                    self.issues.append({
                        'type': 'httpx_mock',
                        'line': decorator.lineno,
                        'target': target,
                        'function': node.name
                    })
                    
        # Check for missing pytest.mark.asyncio on async tests
        if isinstance(node, ast.AsyncFunctionDef):
            has_asyncio_mark = any(
                self.is_asyncio_mark(dec) for dec in node.decorator_list
            )
            if not has_asyncio_mark:
                self.issues.append({
                    'type': 'missing_asyncio_mark',
                    'line': node.lineno,
                    'function': node.name
                })
                
    def is_patch_decorator(self, decorator):
        """Check if decorator is a patch."""
        if isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name) and decorator.func.id == 'patch':
                return True
            if isinstance(decorator.func, ast.Attribute) and decorator.func.attr == 'patch':
                return True
        return False
        
    def is_asyncio_mark(self, decorator):
        """Check if decorator is pytest.mark.asyncio."""
        if isinstance(decorator, ast.Attribute):
            if (isinstance(decorator.value, ast.Attribute) and 
                decorator.value.attr == 'mark' and 
                decorator.attr == 'asyncio'):
                return True
        return False
        
    def get_patch_target(self, decorator):
        """Extract patch target."""
        if isinstance(decorator, ast.Call) and decorator.args:
            if isinstance(decorator.args[0], ast.Constant):
                return decorator.args[0].value
        return None


def analyze_test_file(filepath: Path) -> Dict:
    """Analyze a single test file for async patterns."""
    try:
        content = filepath.read_text()
        tree = ast.parse(content)
        analyzer = AsyncTestModernizer(filepath)
        analyzer.visit(tree)
        
        # Additional regex-based checks
        issues = analyzer.issues.copy()
        
        # Check for AsyncMock with httpx
        async_mock_httpx = re.findall(
            r'AsyncMock.*httpx|mock.*httpx\.AsyncClient', 
            content
        )
        for match in async_mock_httpx:
            issues.append({
                'type': 'async_mock_httpx',
                'pattern': match,
                'file': str(filepath)
            })
            
        # Check for missing async context managers
        if 'async with' not in content and analyzer.async_functions:
            for func in analyzer.async_functions:
                if 'httpx' in content:
                    issues.append({
                        'type': 'missing_async_context',
                        'function': func,
                        'file': str(filepath)
                    })
                    
        return {
            'file': str(filepath),
            'issues': issues,
            'has_respx': analyzer.has_respx_import,
            'has_httpx': analyzer.has_httpx_import,
            'async_tests': len(analyzer.async_functions)
        }
    except Exception as e:
        return {
            'file': str(filepath),
            'error': str(e)
        }


def generate_fixes(analysis_results: List[Dict]) -> Dict[str, List[str]]:
    """Generate fixes for identified issues."""
    fixes = {
        'add_respx_imports': [],
        'replace_httpx_mocks': [],
        'add_asyncio_marks': [],
        'add_async_context_managers': [],
        'create_utilities': []
    }
    
    for result in analysis_results:
        if 'error' in result:
            continue
            
        file = result['file']
        issues = result.get('issues', [])
        
        # Categorize fixes needed
        has_httpx_mock = any(i['type'] in ['httpx_mock', 'async_mock_httpx'] for i in issues)
        needs_respx = has_httpx_mock and not result['has_respx']
        
        if needs_respx:
            fixes['add_respx_imports'].append(file)
            
        if has_httpx_mock:
            fixes['replace_httpx_mocks'].append(file)
            
        for issue in issues:
            if issue['type'] == 'missing_asyncio_mark':
                fixes['add_asyncio_marks'].append(f"{file}:{issue['line']}")
            elif issue['type'] == 'missing_async_context':
                fixes['add_async_context_managers'].append(file)
                
    return fixes


def create_async_test_utilities():
    """Create modern async test utilities."""
    utilities_content = '''"""Modern async test utilities for HTTP testing.

This module provides reusable utilities for async testing with respx.
"""

import asyncio
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

import httpx
import pytest
import respx
from respx.router import MockRouter


class AsyncHTTPTestHelper:
    """Helper for async HTTP testing with respx."""
    
    def __init__(self):
        self.mock_router: Optional[MockRouter] = None
        self.recorded_requests: List[httpx.Request] = []
        
    @asynccontextmanager
    async def mock_http_context(self):
        """Context manager for HTTP mocking."""
        with respx.mock() as mock_router:
            self.mock_router = mock_router
            
            # Add request recording
            @mock_router.route()
            def record_request(request):
                self.recorded_requests.append(request)
                return None  # Let other routes handle the response
            
            yield mock_router
            
    def setup_success_response(
        self, 
        url: str, 
        content: str, 
        status_code: int = 200,
        headers: Optional[Dict[str, str]] = None
    ):
        """Setup a successful HTTP response."""
        if not self.mock_router:
            raise RuntimeError("Must be used within mock_http_context")
            
        response_headers = headers or {"content-type": "text/html"}
        self.mock_router.get(url).mock(
            return_value=httpx.Response(
                status_code,
                text=content,
                headers=response_headers
            )
        )
        
    def setup_error_response(
        self,
        url: str,
        status_code: int = 500,
        error_message: str = "Internal Server Error"
    ):
        """Setup an error HTTP response."""
        if not self.mock_router:
            raise RuntimeError("Must be used within mock_http_context")
            
        self.mock_router.get(url).mock(
            return_value=httpx.Response(
                status_code,
                text=error_message
            )
        )
        
    def setup_timeout(self, url: str):
        """Setup a timeout response."""
        if not self.mock_router:
            raise RuntimeError("Must be used within mock_http_context")
            
        self.mock_router.get(url).mock(
            side_effect=httpx.TimeoutException("Request timed out")
        )
        
    def get_request_count(self, url_pattern: Optional[str] = None) -> int:
        """Get count of requests matching pattern."""
        if not url_pattern:
            return len(self.recorded_requests)
            
        count = 0
        for request in self.recorded_requests:
            if url_pattern in str(request.url):
                count += 1
        return count


@pytest.fixture
async def async_http_helper():
    """Provide async HTTP test helper."""
    return AsyncHTTPTestHelper()


@pytest.fixture
async def mock_successful_scrape(async_http_helper):
    """Mock a successful scrape response."""
    async def _mock_scrape(url: str, content: str):
        async with async_http_helper.mock_http_context():
            async_http_helper.setup_success_response(url, content)
            yield async_http_helper
            
    return _mock_scrape


@pytest.fixture
async def mock_api_responses(async_http_helper):
    """Mock multiple API responses."""
    async def _mock_responses(responses: Dict[str, Dict[str, Any]]):
        async with async_http_helper.mock_http_context():
            for url, response_data in responses.items():
                if response_data.get('error'):
                    async_http_helper.setup_error_response(
                        url,
                        response_data.get('status_code', 500),
                        response_data.get('error')
                    )
                elif response_data.get('timeout'):
                    async_http_helper.setup_timeout(url)
                else:
                    async_http_helper.setup_success_response(
                        url,
                        response_data.get('content', ''),
                        response_data.get('status_code', 200),
                        response_data.get('headers')
                    )
            yield async_http_helper
            
    return _mock_responses
'''
    
    return utilities_content


def generate_fix_report(fixes: Dict[str, List[str]]) -> str:
    """Generate a report of fixes needed."""
    report = ["# Async Test Modernization Report\n"]
    
    total_fixes = sum(len(files) for files in fixes.values())
    report.append(f"Total fixes needed: {total_fixes}\n")
    
    if fixes['add_respx_imports']:
        report.append("\n## Files needing respx imports:")
        for file in fixes['add_respx_imports']:
            report.append(f"- {file}")
            
    if fixes['replace_httpx_mocks']:
        report.append("\n## Files with httpx mocks to replace:")
        for file in fixes['replace_httpx_mocks']:
            report.append(f"- {file}")
            
    if fixes['add_asyncio_marks']:
        report.append("\n## Missing @pytest.mark.asyncio decorators:")
        for location in fixes['add_asyncio_marks']:
            report.append(f"- {location}")
            
    if fixes['add_async_context_managers']:
        report.append("\n## Files needing async context managers:")
        for file in fixes['add_async_context_managers']:
            report.append(f"- {file}")
            
    report.append("\n## Recommended Actions:")
    report.append("1. Add respx to test dependencies if not already present")
    report.append("2. Replace httpx mocks with respx patterns")
    report.append("3. Ensure all async tests have @pytest.mark.asyncio")
    report.append("4. Use async context managers for resource management")
    report.append("5. Create and use the async test utilities")
    
    return "\n".join(report)


@click.command()
@click.option(
    '--fix', 
    is_flag=True, 
    help='Apply fixes automatically where possible'
)
@click.option(
    '--create-utilities',
    is_flag=True,
    help='Create async test utilities file'
)
def main(fix: bool, create_utilities: bool):
    """Modernize async testing patterns."""
    test_dir = Path("tests")
    
    # Find all test files
    test_files = list(test_dir.rglob("test_*.py"))
    
    click.echo(f"Analyzing {len(test_files)} test files...")
    
    # Analyze each file
    results = []
    for filepath in test_files:
        result = analyze_test_file(filepath)
        if result.get('issues'):
            results.append(result)
            
    # Generate fixes
    fixes = generate_fixes(results)
    
    # Generate report
    report = generate_fix_report(fixes)
    click.echo(report)
    
    # Save report
    report_path = Path("async_test_modernization_report.md")
    report_path.write_text(report)
    click.echo(f"\nReport saved to: {report_path}")
    
    # Create utilities if requested
    if create_utilities:
        utilities_path = Path("tests/fixtures/async_http_utilities.py")
        utilities_path.write_text(create_async_test_utilities())
        click.echo(f"Created async test utilities at: {utilities_path}")
        
    if fix:
        click.echo("\nApplying automatic fixes...")
        # Run ruff to add missing imports and fix formatting
        subprocess.run(["ruff", "check", "tests", "--fix"], check=False)
        subprocess.run(["ruff", "format", "tests"], check=False)
        click.echo("Automatic fixes applied. Manual review still required for:")
        click.echo("- Replacing httpx mocks with respx patterns")
        click.echo("- Adding async context managers")
        click.echo("- Updating test logic for modern patterns")


if __name__ == "__main__":
    main()