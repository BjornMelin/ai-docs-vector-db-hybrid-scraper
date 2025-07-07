#!/usr/bin/env python3
"""Fix the observability test to properly handle global variables."""

import re
from pathlib import Path


def fix_observability_globals():
    """Fix the test to properly set global variables for shutdown."""
    test_file = Path("tests/integration/test_observability_integration.py")
    
    if not test_file.exists():
        print(f"Test file not found: {test_file}")
        return
        
    content = test_file.read_text()
    
    # Find and update the test method
    test_method_pattern = r'(async def test_complete_observability_flow\([^)]+\):\s*"""[^"]+""")'
    
    new_test_setup = '''async def test_complete_observability_flow(
        self,
        _mock_auto_instrumentation,
        mock_resource,
        mock_metric_reader,
        _mock_metric_exporter,
        _mock_span_exporter,
        mock_span_processor,
        mock_meter_provider,
        mock_tracer_provider,
        _mock_metrics,
        _mock_trace,
    ):
        """Test complete observability flow from configuration to shutdown."""
        # Import the module to access globals
        from src.services.observability import init as obs_init'''
    
    # Replace the test method signature
    content = re.sub(test_method_pattern, new_test_setup, content, flags=re.DOTALL)
    
    # Now fix the part where we need to set the globals
    setup_end_pattern = r'(mock_meter_provider_instance\.get_meter\.return_value = mock_meter)'
    
    globals_setup = '''mock_meter_provider_instance.get_meter.return_value = mock_meter
        
        # Set the global variables that shutdown_observability expects
        obs_init._tracer_provider = mock_tracer_provider_instance
        obs_init._meter_provider = mock_meter_provider_instance'''
    
    content = re.sub(setup_end_pattern, globals_setup, content)
    
    # Fix the shutdown assertions to check is_observability_enabled after clearing globals
    shutdown_pattern = r'(# Step 5: Test shutdown\s*\n\s*shutdown_observability\(\)\s*\n\s*assert is_observability_enabled\(\) is False)'
    
    new_shutdown_check = '''# Step 5: Test shutdown
            shutdown_observability()
            
            # After shutdown, the globals should be None
            assert obs_init._tracer_provider is None
            assert obs_init._meter_provider is None
            assert is_observability_enabled() is False'''
    
    content = re.sub(shutdown_pattern, new_shutdown_check, content)
    
    # Write the fixed content
    test_file.write_text(content)
    print(f"Fixed observability globals handling in {test_file}")


if __name__ == "__main__":
    fix_observability_globals()