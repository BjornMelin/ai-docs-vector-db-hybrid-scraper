"""Property-based testing fixtures and configuration."""

import pytest
from hypothesis import settings, Verbosity

# Configure Hypothesis settings for property-based tests
@pytest.fixture(autouse=True)
def hypothesis_settings():
    """Configure Hypothesis settings for all property-based tests."""
    # Increase test cases for thorough testing
    settings.register_profile("property_testing", max_examples=1000, verbosity=Verbosity.verbose)
    settings.register_profile("ci", max_examples=500)
    settings.register_profile("dev", max_examples=100)
    
    # Use appropriate profile based on environment
    import os
    profile = os.getenv("HYPOTHESIS_PROFILE", "dev")
    settings.load_profile(profile)

@pytest.fixture
def hypothesis_settings_debug():
    """Enhanced Hypothesis settings for debugging test failures."""
    settings.register_profile("debug", max_examples=10, verbosity=Verbosity.debug)
    settings.load_profile("debug")