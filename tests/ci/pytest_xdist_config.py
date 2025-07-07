"""pytest-xdist optimization configuration for CI/CD environments.

This module provides optimized pytest-xdist configuration for different
CI platforms and execution environments with performance monitoring.
"""

import os
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import psutil


@dataclass
class XDistConfig:
    """Configuration for pytest-xdist execution."""

    # Worker configuration
    num_workers: int = field(default_factory=lambda: 1)
    dist_mode: str = "loadscope"  # loadscope, loadfile, loadgroup, or no
    max_workers: int = 4

    # Timeout configuration
    timeout: int = 300
    timeout_method: str = "thread"

    # Resource limits
    max_memory_per_worker_mb: int = 1024
    cpu_affinity: bool = False

    # Test distribution
    test_group_strategy: str = "module"  # module, class, or function
    rerun_failed: int = 0

    # Performance monitoring
    collect_performance_metrics: bool = True
    generate_timing_report: bool = True

    # Platform-specific
    platform_optimizations: dict[str, Any] = field(default_factory=dict)


class CIEnvironmentDetector:
    """Detects and configures for different CI environments."""

    @staticmethod
    def detect() -> dict[str, Any]:
        """Detect CI environment and capabilities."""
        env_info = {
            "is_ci": bool(os.getenv("CI")),
            "platform": platform.system().lower(),
            "cpu_count": os.cpu_count() or 1,
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "ci_provider": None,
            "runner_specs": {},
        }

        # Detect CI provider
        if os.getenv("GITHUB_ACTIONS"):
            env_info["ci_provider"] = "github"
            env_info["runner_specs"] = CIEnvironmentDetector._get_github_runner_specs()
        elif os.getenv("GITLAB_CI"):
            env_info["ci_provider"] = "gitlab"
            env_info["runner_specs"] = CIEnvironmentDetector._get_gitlab_runner_specs()
        elif os.getenv("JENKINS_URL"):
            env_info["ci_provider"] = "jenkins"
            env_info["runner_specs"] = CIEnvironmentDetector._get_jenkins_specs()
        elif os.getenv("SYSTEM_TEAMFOUNDATIONCOLLECTIONURI"):
            env_info["ci_provider"] = "azure_devops"
            env_info["runner_specs"] = CIEnvironmentDetector._get_azure_specs()
        elif os.getenv("CIRCLECI"):
            env_info["ci_provider"] = "circleci"
            env_info["runner_specs"] = CIEnvironmentDetector._get_circleci_specs()

        return env_info

    @staticmethod
    def _get_github_runner_specs() -> dict[str, Any]:
        """Get GitHub Actions runner specifications."""
        runner_os = os.getenv("RUNNER_OS", "").lower()
        runner_arch = os.getenv("RUNNER_ARCH", "X64")

        # Standard GitHub-hosted runner specs
        specs = {
            "ubuntu-latest": {"cores": 2, "memory_gb": 7, "storage_gb": 14},
            "windows-latest": {"cores": 2, "memory_gb": 7, "storage_gb": 14},
            "macos-latest": {"cores": 3, "memory_gb": 14, "storage_gb": 14},
            "macos-13": {"cores": 3, "memory_gb": 14, "storage_gb": 14},
            "ubuntu-22.04": {"cores": 2, "memory_gb": 7, "storage_gb": 14},
            "ubuntu-20.04": {"cores": 2, "memory_gb": 7, "storage_gb": 14},
        }

        runner_image = os.getenv("ImageOS", "ubuntu-latest")
        return specs.get(runner_image, specs["ubuntu-latest"])

    @staticmethod
    def _get_gitlab_runner_specs() -> dict[str, Any]:
        """Get GitLab CI runner specifications."""
        return {
            "cores": int(os.getenv("CI_RUNNER_EXECUTABLE_ARCH_CPUS", "2")),
            "memory_gb": 4,  # Default GitLab shared runner
            "storage_gb": 25,
        }

    @staticmethod
    def _get_jenkins_specs() -> dict[str, Any]:
        """Get Jenkins runner specifications."""
        return {
            "cores": os.cpu_count() or 2,
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "storage_gb": 50,  # Typical Jenkins agent
        }

    @staticmethod
    def _get_azure_specs() -> dict[str, Any]:
        """Get Azure DevOps runner specifications."""
        return {
            "cores": int(os.getenv("NUMBER_OF_PROCESSORS", "2")),
            "memory_gb": 7,  # Standard Azure hosted agent
            "storage_gb": 10,
        }

    @staticmethod
    def _get_circleci_specs() -> dict[str, Any]:
        """Get CircleCI runner specifications."""
        resource_class = os.getenv("CIRCLE_JOB_RESOURCE_CLASS", "medium")

        specs = {
            "small": {"cores": 1, "memory_gb": 2},
            "medium": {"cores": 2, "memory_gb": 4},
            "medium+": {"cores": 3, "memory_gb": 6},
            "large": {"cores": 4, "memory_gb": 8},
            "xlarge": {"cores": 8, "memory_gb": 16},
            "2xlarge": {"cores": 16, "memory_gb": 32},
        }

        return specs.get(resource_class, specs["medium"])


class XDistOptimizer:
    """Optimizes pytest-xdist configuration for the environment."""

    def __init__(self):
        self.env_info = CIEnvironmentDetector.detect()

    def get_optimal_config(self) -> XDistConfig:
        """Get optimal xdist configuration for current environment."""
        config = XDistConfig()

        # Base configuration on environment
        if self.env_info["is_ci"]:
            config = self._configure_for_ci(config)
        else:
            config = self._configure_for_local(config)

        # Apply platform-specific optimizations
        config = self._apply_platform_optimizations(config)

        # Apply safety limits
        config = self._apply_safety_limits(config)

        return config

    def _configure_for_ci(self, config: XDistConfig) -> XDistConfig:
        """Configure for CI environment."""
        ci_provider = self.env_info["ci_provider"]
        runner_specs = self.env_info["runner_specs"]

        if ci_provider == "github":
            # GitHub Actions optimization
            config.num_workers = min(runner_specs["cores"], 4)
            config.max_workers = 4
            config.timeout = 300
            config.max_memory_per_worker_mb = int(
                (runner_specs["memory_gb"] * 1024) / (config.num_workers + 1)
            )
            config.dist_mode = "loadscope"

        elif ci_provider == "gitlab":
            # GitLab CI optimization
            config.num_workers = min(runner_specs["cores"], 2)
            config.max_workers = 2
            config.timeout = 600
            config.max_memory_per_worker_mb = 1024
            config.dist_mode = "loadfile"

        elif ci_provider == "jenkins":
            # Jenkins optimization
            config.num_workers = max(1, runner_specs["cores"] // 2)
            config.max_workers = 8
            config.timeout = 900
            config.dist_mode = "loadscope"

        elif ci_provider == "azure_devops":
            # Azure DevOps optimization
            config.num_workers = min(runner_specs["cores"], 4)
            config.max_workers = 4
            config.timeout = 600
            config.dist_mode = "loadscope"

        elif ci_provider == "circleci":
            # CircleCI optimization
            config.num_workers = runner_specs["cores"]
            config.max_workers = runner_specs["cores"]
            config.timeout = 300
            config.max_memory_per_worker_mb = int(
                (runner_specs["memory_gb"] * 1024 * 0.8) / config.num_workers
            )
            config.dist_mode = "loadscope"

        else:
            # Generic CI optimization
            config.num_workers = min(self.env_info["cpu_count"], 4)
            config.max_workers = 4
            config.timeout = 600
            config.dist_mode = "loadscope"

        return config

    def _configure_for_local(self, config: XDistConfig) -> XDistConfig:
        """Configure for local development."""
        cpu_count = self.env_info["cpu_count"]
        memory_gb = self.env_info["memory_gb"]

        # Conservative local settings
        config.num_workers = min(max(1, cpu_count // 2), 2)
        config.max_workers = 2
        config.timeout = 300
        config.max_memory_per_worker_mb = int((memory_gb * 1024 * 0.5) / config.num_workers)
        config.dist_mode = "loadscope"
        config.collect_performance_metrics = True

        return config

    def _apply_platform_optimizations(self, config: XDistConfig) -> XDistConfig:
        """Apply platform-specific optimizations."""
        platform_name = self.env_info["platform"]

        if platform_name == "linux":
            config.platform_optimizations = {
                "use_process_groups": True,
                "enable_cpu_affinity": self.env_info["is_ci"],
                "memory_limit_enforcement": True,
            }

        elif platform_name == "darwin":  # macOS
            config.platform_optimizations = {
                "use_spawn_method": True,
                "disable_acceleration_framework": True,
                "memory_pressure_monitoring": True,
            }
            # macOS typically has better single-thread performance
            config.num_workers = min(config.num_workers, 3)

        elif platform_name == "windows":
            config.platform_optimizations = {
                "use_spawn_method": True,
                "process_priority": "below_normal",
                "handle_limit": 2048,
            }
            # Windows has higher process overhead
            config.num_workers = min(config.num_workers, 2)
            config.timeout_method = "thread"  # More reliable on Windows

        return config

    def _apply_safety_limits(self, config: XDistConfig) -> XDistConfig:
        """Apply safety limits to prevent resource exhaustion."""
        # Ensure we leave resources for the system
        total_memory_mb = self.env_info["memory_gb"] * 1024
        max_test_memory = total_memory_mb * 0.8  # Use max 80% of total memory

        config.max_memory_per_worker_mb = min(
            config.max_memory_per_worker_mb,
            int(max_test_memory / config.num_workers)
        )

        # Ensure reasonable worker count
        config.num_workers = min(config.num_workers, config.max_workers)
        config.num_workers = max(1, config.num_workers)

        return config


def get_xdist_args(custom_config: XDistConfig | None = None) -> list[str]:
    """Get pytest-xdist command line arguments."""
    if custom_config is None:
        optimizer = XDistOptimizer()
        config = optimizer.get_optimal_config()
    else:
        config = custom_config

    args = [
        f"--numprocesses={config.num_workers}",
        f"--dist={config.dist_mode}",
        f"--maxprocesses={config.max_workers}",
        f"--timeout={config.timeout}",
        f"--timeout-method={config.timeout_method}",
    ]

    if config.rerun_failed > 0:
        args.append(f"--reruns={config.rerun_failed}")

    # Add platform-specific arguments
    if config.platform_optimizations.get("use_process_groups"):
        args.append("--tx=popen//group")
    elif config.platform_optimizations.get("use_spawn_method"):
        args.append("--tx=popen//python=python")

    return args


def print_xdist_config():
    """Print the current xdist configuration (useful for debugging)."""
    optimizer = XDistOptimizer()
    config = optimizer.get_optimal_config()
    env_info = optimizer.env_info

    print("pytest-xdist Configuration")
    print("=" * 50)
    print(f"Environment: {'CI' if env_info['is_ci'] else 'Local'}")
    if env_info["ci_provider"]:
        print(f"CI Provider: {env_info['ci_provider']}")
    print(f"Platform: {env_info['platform']}")
    print(f"CPU Count: {env_info['cpu_count']}")
    print(f"Memory: {env_info['memory_gb']:.1f} GB")
    print()
    print("Configuration:")
    print(f"  Workers: {config.num_workers} (max: {config.max_workers})")
    print(f"  Distribution: {config.dist_mode}")
    print(f"  Timeout: {config.timeout}s ({config.timeout_method})")
    print(f"  Memory/Worker: {config.max_memory_per_worker_mb} MB")
    print()
    print("Command line args:")
    print(" ".join(get_xdist_args(config)))


if __name__ == "__main__":
    # Print configuration when run directly
    print_xdist_config()
