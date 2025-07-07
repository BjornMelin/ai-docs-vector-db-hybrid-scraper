"""Tests for federated search service.

This file imports tests from split modules to maintain organization while keeping file sizes manageable.
"""

import pytest

# Import all test classes from split files
from .test_federated_enums import TestEnums
from .test_federated_integration import TestSearchModeIntegration
from .test_federated_models import (
    TestCollectionMetadata,
    TestCollectionSearchResult,
    TestFederatedSearchRequest,
    TestFederatedSearchResult,
)
from .test_federated_service_advanced import TestFederatedSearchServiceAdvanced
from .test_federated_service_core import TestFederatedSearchServiceCore
from .test_federated_service_merging import TestFederatedSearchServiceMerging


# Re-export all test classes for pytest discovery
__all__ = [
    "TestCollectionMetadata",
    "TestCollectionSearchResult",
    "TestEnums",
    "TestFederatedSearchRequest",
    "TestFederatedSearchResult",
    "TestFederatedSearchServiceAdvanced",
    "TestFederatedSearchServiceCore",
    "TestFederatedSearchServiceMerging",
    "TestSearchModeIntegration",
]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
