#!/usr/bin/env python3
"""Fix the observability test flow to properly handle initialization."""

import re
from pathlib import Path


def fix_observability_test():
    """Fix the test_complete_observability_flow test."""
    test_file = Path("tests/integration/test_observability_integration.py")
    
    if not test_file.exists():
        print(f"Test file not found: {test_file}")
        return
        
    content = test_file.read_text()
    
    # Fix 1: Add initialization patch to ensure _tracer_provider is set
    init_patch_fix = '''@patch("src.services.observability.init._tracer_provider", new=MagicMock())
    @patch("src.services.observability.init._meter_provider", new=MagicMock())'''
    
    # Find the patch decorators for test_complete_observability_flow
    pattern = r'(@patch\("src\.services\.observability\.init\.trace"\)\s*\n\s*@patch\("src\.services\.observability\.init\.metrics"\))'
    
    if re.search(pattern, content):
        # Add the new patches before the existing ones
        content = re.sub(
            pattern,
            init_patch_fix + '\n    ' + r'\1',
            content
        )
        print("✓ Added _tracer_provider and _meter_provider patches")
    
    # Fix 2: Ensure the mocked TracerProvider and MeterProvider are configured correctly
    tracer_setup_fix = '''        # Setup comprehensive mocks
        mock_resource.create.return_value = MagicMock()

        mock_tracer_provider_instance = MagicMock()
        mock_tracer_provider.return_value = mock_tracer_provider_instance

        mock_meter_provider_instance = MagicMock()
        mock_meter_provider.return_value = mock_meter_provider_instance

        mock_span_processor_instance = MagicMock()
        mock_span_processor.return_value = mock_span_processor_instance

        mock_metric_reader_instance = MagicMock()
        mock_metric_reader.return_value = mock_metric_reader_instance
        
        # Mock the get_tracer_provider and get_meter_provider functions
        _mock_trace.get_tracer_provider.return_value = mock_tracer_provider_instance
        _mock_metrics.get_meter_provider.return_value = mock_meter_provider_instance
        
        # Mock tracer and meter instances
        mock_tracer = MagicMock()
        mock_meter = MagicMock()
        mock_tracer_provider_instance.get_tracer.return_value = mock_tracer
        mock_meter_provider_instance.get_meter.return_value = mock_meter'''
    
    # Find the existing setup and replace it
    setup_pattern = r'(# Setup comprehensive mocks[\s\S]*?mock_metric_reader\.return_value = mock_metric_reader_instance)'
    
    if re.search(setup_pattern, content):
        content = re.sub(setup_pattern, tracer_setup_fix, content)
        print("✓ Fixed mock setup for TracerProvider and MeterProvider")
    
    # Fix 3: Import MagicMock if not already imported
    if "from unittest.mock import MagicMock" not in content:
        import_pattern = r'(from unittest\.mock import[^\n]+)'
        if re.search(import_pattern, content):
            content = re.sub(
                import_pattern,
                r'\1, MagicMock',
                content
            )
        else:
            # Add import at the top with other imports
            import_insert = "from unittest.mock import MagicMock, patch\n"
            content = re.sub(
                r'(import pytest\n)',
                r'\1' + import_insert,
                content
            )
        print("✓ Added MagicMock import")
    
    # Write the fixed content
    test_file.write_text(content)
    print(f"\nFixed observability test in {test_file}")
    print("Run the test again to verify the fix.")


if __name__ == "__main__":
    fix_observability_test()