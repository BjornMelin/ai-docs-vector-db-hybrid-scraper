#!/usr/bin/env python3
"""Fix function-level imports in client_manager.py by replacing them with None checks."""

import re


# Read the file
with open("src/infrastructure/client_manager.py") as f:
    content = f.read()

# Define the replacements
replacements = [
    (
        r"from src\.services\.cache\.manager import \(\s*CacheManager,\s*\)",
        "if CacheManager is None:\n                        raise ImportError(\"CacheManager not available\")"
    ),
    (
        r"from src\.services\.crawling\.manager import \(\s*CrawlingManager,\s*\)",
        "if CrawlingManager is None:\n                        raise ImportError(\"CrawlingManager not available\")"
    ),
    (
        r"from src\.services\.hyde\.config import \(\s*HyDEConfig,\s*create_default_hyde_config,\s*\)",
        "if HyDEConfig is None or create_default_hyde_config is None:\n                        raise ImportError(\"HyDE components not available\")"
    ),
    (
        r"from src\.services\.core\.project_storage import \(\s*ProjectStorageService,\s*\)",
        "if ProjectStorageService is None:\n                        raise ImportError(\"ProjectStorageService not available\")"
    ),
    (
        r"from src\.services\.browser\.browser_router import \(\s*EnhancedAutomationRouter,\s*\)",
        "if EnhancedAutomationRouter is None:\n                        raise ImportError(\"EnhancedAutomationRouter not available\")"
    ),
    (
        r"from src\.services\.task_queue\.manager import \(\s*TaskQueueManager,\s*\)",
        "if TaskQueueManager is None:\n                        raise ImportError(\"TaskQueueManager not available\")"
    ),
    (
        r"from src\.services\.content_intelligence\.service import \(\s*ContentIntelligenceService,\s*\)",
        "if ContentIntelligenceService is None:\n                        raise ImportError(\"ContentIntelligenceService not available\")"
    ),
    (
        r"from src\.services\.rag import RAGGenerator",
        "if RAGGenerator is None:\n                        raise ImportError(\"RAGGenerator not available\")"
    ),
    (
        r"from src\.services\.query_processing import \(\s*QueryProcessor,\s*\)",
        "if QueryProcessor is None:\n                        raise ImportError(\"QueryProcessor not available\")"
    ),
    (
        r"from src\.services\.deployment\.feature_flags import \(\s*FeatureFlagManager,\s*\)",
        "if FeatureFlagManager is None:\n                        raise ImportError(\"FeatureFlagManager not available\")"
    ),
    (
        r"from src\.services\.deployment\.ab_testing import \(\s*ABTestingManager,\s*\)",
        "if ABTestingManager is None:\n                        raise ImportError(\"ABTestingManager not available\")"
    ),
    (
        r"from src\.services\.deployment\.blue_green import \(\s*BlueGreenDeploymentManager,\s*\)",
        "if BlueGreenDeploymentManager is None:\n                        raise ImportError(\"BlueGreenDeploymentManager not available\")"
    ),
    (
        r"from src\.services\.deployment\.canary import \(\s*CanaryDeploymentManager,\s*\)",
        "if CanaryDeploymentManager is None:\n                        raise ImportError(\"CanaryDeploymentManager not available\")"
    ),
]

# Apply replacements
for pattern, replacement in replacements:
    content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)

# Write back to file
with open("src/infrastructure/client_manager.py", "w") as f:
    f.write(content)

print("Fixed client_manager.py imports")
