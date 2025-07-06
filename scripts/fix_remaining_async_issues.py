#!/usr/bin/env python3
"""Fix remaining async test issues in specific files."""

import re
from pathlib import Path


def fix_test_tracking_file(file_path: Path) -> None:
    """Fix event loop issues in test_tracking.py."""
    content = file_path.read_text()
    
    # Replace the problematic async test pattern
    pattern = r'''@instrument_function\(operation_type="test_operation"\)
        @pytest\.mark\.asyncio
        async def test_async_function\(arg1, arg2=None\):
            return f"result-\{arg1\}-\{arg2\}"

        # Test function execution
        loop = asyncio\.new_event_loop\(\)
        asyncio\.set_event_loop\(loop\)
        try:
            result = loop\.run_until_complete\(
                test_async_function\("value1", arg2="value2"\)
            \)

            assert result == "result-value1-value2"

            # Verify span attributes
            mock_span\.set_attribute\.assert_any_call\("operation\.type", "test_operation"\)
            mock_span\.set_attribute\.assert_any_call\(
                "function\.name", "test_async_function"
            \)
            mock_span\.set_attribute\.assert_any_call\("function\.success", True\)

        finally:
            loop\.close\(\)'''
    
    replacement = '''@instrument_function(operation_type="test_operation")
        async def test_async_function(arg1, arg2=None):
            return f"result-{arg1}-{arg2}"

        # Test function execution
        result = await test_async_function("value1", arg2="value2")

        assert result == "result-value1-value2"

        # Verify span attributes
        mock_span.set_attribute.assert_any_call("operation.type", "test_operation")
        mock_span.set_attribute.assert_any_call(
            "function.name", "test_async_function"
        )
        mock_span.set_attribute.assert_any_call("function.success", True)'''
    
    content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)
    
    # Also ensure the test method has @pytest.mark.asyncio
    content = re.sub(
        r'(def test_instrument_async_function\(self, mock_get_tracer\):)',
        r'@pytest.mark.asyncio\n    async \1',
        content
    )
    
    file_path.write_text(content)
    print(f"Fixed {file_path}")


def fix_observability_integration_file(file_path: Path) -> None:
    """Fix event loop issues in test_observability_integration.py."""
    content = file_path.read_text()
    
    # Fix the health check test with event loop
    pattern = r'''# Step 4: Test health check
            loop = asyncio\.new_event_loop\(\)
            asyncio\.set_event_loop\(loop\)
            try:
                health = loop\.run_until_complete\(get_observability_health\(service\)\)

                assert health\["enabled"\] is True
                assert health\["status"\] == "healthy"

            finally:
                loop\.close\(\)'''
    
    replacement = '''# Step 4: Test health check
            health = await get_observability_health(service)

            assert health["enabled"] is True
            assert health["status"] == "healthy"'''
    
    content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)
    
    # Ensure the test method is async
    content = re.sub(
        r'(def test_complete_observability_flow\([^)]+\):)',
        r'@pytest.mark.asyncio\n    async \1',
        content
    )
    
    file_path.write_text(content)
    print(f"Fixed {file_path}")


def fix_mcp_services_integration_file(file_path: Path) -> None:
    """Fix event loop and timing issues in test_mcp_services_integration.py."""
    content = file_path.read_text()
    
    # Fix missing start_time variable
    pattern = r'# Verify performance \(should complete quickly\)\n        execution_time = end_time - start_time'
    replacement = '''# Run concurrent operations on all services
        start_time = time.time()
        tasks = []
        for service in services.values():
            task = asyncio.create_task(get_service_info_concurrent(service))
            tasks.append(task)

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()

        # Verify all operations succeeded
        for result in results:
            assert not isinstance(result, Exception)
            assert "service" in result
            assert result["status"] == "active"

        # Verify performance (should complete quickly)
        execution_time = end_time - start_time'''
    
    content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    file_path.write_text(content)
    print(f"Fixed {file_path}")


def main():
    """Main function to fix remaining async issues."""
    base_path = Path("/workspace/repos/ai-docs-vector-db-hybrid-scraper")
    
    # Fix specific files with event loop issues
    files_to_fix = [
        (base_path / "tests/unit/services/observability/test_tracking.py", fix_test_tracking_file),
        (base_path / "tests/integration/test_observability_integration.py", fix_observability_integration_file),
        (base_path / "tests/integration/mcp_services/test_mcp_services_integration.py", fix_mcp_services_integration_file),
    ]
    
    for file_path, fix_function in files_to_fix:
        if file_path.exists():
            try:
                fix_function(file_path)
            except Exception as e:
                print(f"Error fixing {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")


if __name__ == "__main__":
    main()