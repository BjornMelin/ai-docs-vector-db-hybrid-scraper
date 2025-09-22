"""Test environment configurations for common CI platforms."""

import os


class TestEnvironment:
    """Base class for test environment configuration."""

    def __init__(self):
        self.name = self.__class__.__name__
        self.is_active = self.detect()

    def detect(self) -> bool:
        """Detect if this environment is active."""
        raise NotImplementedError

    def get_pytest_args(self) -> list[str]:
        """Get environment-specific pytest arguments."""
        return []

    def get_env_vars(self) -> dict[str, str]:
        """Get environment-specific environment variables."""
        return {}

    def get_test_selection(self) -> tuple[str, list[str]]:
        """Get test marker expression and test paths."""
        return "", ["tests"]


class GitHubActionsEnvironment(TestEnvironment):
    """GitHub Actions CI environment configuration."""

    def detect(self) -> bool:
        return bool(os.getenv("GITHUB_ACTIONS"))

    def get_pytest_args(self) -> list[str]:
        runner_os = os.getenv("RUNNER_OS", "Linux").lower()

        # Base args for all GitHub runners
        args = [
            "--numprocesses=auto",
            "--dist=loadscope",
            "--maxprocesses=4",
            "--max-worker-restart=1",
            f"--timeout={300 if runner_os == 'windows' else 180}",
            "--timeout-method=thread",
            "--tb=short",
            "--no-header",
            "--color=yes",
            "-v",
        ]

        # OS-specific optimizations
        if runner_os == "windows":
            args.extend(
                [
                    "--numprocesses=2",  # Windows has higher process overhead
                    "--timeout=300",
                ]
            )
        elif runner_os == "macos":
            args.extend(
                [
                    "--numprocesses=3",  # macOS runners have 3 cores
                ]
            )

        # Add GitHub-specific output formatting
        args.extend(
            [
                "--junit-xml=test-results/junit.xml",
                "--html=test-results/report.html",
                "--self-contained-html",
            ]
        )

        return args

    def get_env_vars(self) -> dict[str, str]:
        return {
            "PYTEST_XDIST_AUTO_NUM_WORKERS": "4",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "PIP_NO_CACHE_DIR": "1",
            "COVERAGE_PARALLEL": "true",
        }

    def get_test_selection(self) -> tuple[str, list[str]]:
        # Run different test sets based on event type
        event_name = os.getenv("GITHUB_EVENT_NAME", "push")

        if event_name == "pull_request":
            # Fast feedback for PRs
            return "not slow and not integration", ["tests/unit"]
        if event_name == "push":
            # More comprehensive for pushes to main
            return "not slow", ["tests/unit", "tests/integration"]
        # Full test suite for scheduled runs
        return "", ["tests"]


class GitLabCIEnvironment(TestEnvironment):
    """GitLab CI environment configuration."""

    def detect(self) -> bool:
        return bool(os.getenv("GITLAB_CI"))

    def get_pytest_args(self) -> list[str]:
        # GitLab runners typically have fewer resources
        return [
            "--numprocesses=2",
            "--dist=loadfile",  # Better for GitLab's shared runners
            "--maxprocesses=2",
            "--max-worker-restart=2",
            "--timeout=600",  # Longer timeout for potentially slower runners
            "--timeout-method=thread",
            "--tb=short",
            "--junit-xml=report.xml",
            "-v",
        ]

    def get_env_vars(self) -> dict[str, str]:
        return {
            "PYTEST_XDIST_AUTO_NUM_WORKERS": "2",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "COVERAGE_PARALLEL": "true",
        }

    def get_test_selection(self) -> tuple[str, list[str]]:
        ci_pipeline_source = os.getenv("CI_PIPELINE_SOURCE", "push")

        if ci_pipeline_source == "merge_request_event":
            return "not slow and not integration", ["tests/unit"]
        return "not slow", ["tests"]


class JenkinsEnvironment(TestEnvironment):
    """Jenkins CI environment configuration."""

    def detect(self) -> bool:
        return bool(os.getenv("JENKINS_URL"))

    def get_pytest_args(self) -> list[str]:
        # Jenkins typically has more resources available
        cpu_count = os.cpu_count() or 2
        num_workers = min(cpu_count - 1, 8)

        return [
            f"--numprocesses={num_workers}",
            "--dist=loadscope",
            f"--maxprocesses={num_workers}",
            "--max-worker-restart=3",
            "--timeout=900",
            "--timeout-method=thread",
            "--tb=short",
            "--junitxml=test-results.xml",
            "-v",
        ]

    def get_env_vars(self) -> dict[str, str]:
        return {
            "PYTEST_XDIST_AUTO_NUM_WORKERS": str(min(os.cpu_count() or 2, 8)),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        }


class AzureDevOpsEnvironment(TestEnvironment):
    """Azure DevOps environment configuration."""

    def detect(self) -> bool:
        return bool(os.getenv("SYSTEM_TEAMFOUNDATIONCOLLECTIONURI"))

    def get_pytest_args(self) -> list[str]:
        return [
            "--numprocesses=auto",
            "--dist=loadscope",
            "--maxprocesses=4",
            "--max-worker-restart=2",
            "--timeout=300",
            "--timeout-method=thread",
            "--tb=short",
            "--junit-xml=test-results.xml",
            "--cov-report=cobertura:coverage.xml",  # Azure DevOps prefers Cobertura
            "-v",
        ]

    def get_env_vars(self) -> dict[str, str]:
        return {
            "PYTEST_XDIST_AUTO_NUM_WORKERS": "4",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "COVERAGE_PARALLEL": "true",
        }


class CircleCIEnvironment(TestEnvironment):
    """CircleCI environment configuration."""

    def detect(self) -> bool:
        return bool(os.getenv("CIRCLECI"))

    def get_pytest_args(self) -> list[str]:
        # CircleCI resource class determines available resources
        resource_class = os.getenv("CIRCLE_JOB_RESOURCE_CLASS", "medium")

        resource_configs = {
            "small": {"workers": 1, "timeout": 300},
            "medium": {"workers": 2, "timeout": 300},
            "medium+": {"workers": 3, "timeout": 300},
            "large": {"workers": 4, "timeout": 600},
            "xlarge": {"workers": 8, "timeout": 900},
            "2xlarge": {"workers": 16, "timeout": 1200},
        }

        config = resource_configs.get(resource_class, resource_configs["medium"])

        return [
            f"--numprocesses={config['workers']}",
            "--dist=loadscope",
            f"--maxprocesses={config['workers']}",
            "--max-worker-restart=2",
            f"--timeout={config['timeout']}",
            "--timeout-method=thread",
            "--tb=short",
            "--junit-xml=$CIRCLE_TEST_REPORTS/pytest/junit.xml",
            "-v",
        ]

    def get_env_vars(self) -> dict[str, str]:
        return {
            "PYTEST_XDIST_AUTO_NUM_WORKERS": "4",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        }


class LocalEnvironment(TestEnvironment):
    """Local development environment configuration."""

    def detect(self) -> bool:
        # Active if no CI environment is detected
        ci_envs = [
            "CI",
            "GITHUB_ACTIONS",
            "GITLAB_CI",
            "JENKINS_URL",
            "SYSTEM_TEAMFOUNDATIONCOLLECTIONURI",
            "CIRCLECI",
        ]
        return not any(os.getenv(var) for var in ci_envs)

    def get_pytest_args(self) -> list[str]:
        # Conservative settings for local development
        return [
            "--numprocesses=2",  # Don't overwhelm developer machine
            "--dist=loadscope",
            "--maxprocesses=2",
            "--timeout=300",
            "--timeout-method=thread",
            "--tb=short",
            "--durations=10",
            "-v",
        ]

    def get_env_vars(self) -> dict[str, str]:
        return {
            "PYTEST_XDIST_AUTO_NUM_WORKERS": "2",
            "PYTHONDONTWRITEBYTECODE": "1",
        }

    def get_test_selection(self) -> tuple[str, list[str]]:
        # Allow running all tests locally
        return "", ["tests"]


def detect_environment() -> TestEnvironment:
    """Detect the current test environment."""
    environments = [
        GitHubActionsEnvironment(),
        GitLabCIEnvironment(),
        JenkinsEnvironment(),
        AzureDevOpsEnvironment(),
        CircleCIEnvironment(),
        LocalEnvironment(),
    ]

    for env in environments:
        if env.is_active:
            return env

    # Fallback to local environment
    return LocalEnvironment()


def get_optimized_pytest_command() -> str:
    """Get the optimized pytest command for the current environment."""
    env = detect_environment()

    # Base command
    cmd_parts = ["pytest"]

    # Add environment-specific arguments
    cmd_parts.extend(env.get_pytest_args())

    # Add test selection
    marker_expr, test_paths = env.get_test_selection()
    if marker_expr:
        cmd_parts.extend(["-m", f'"{marker_expr}"'])
    cmd_parts.extend(test_paths)

    return " ".join(cmd_parts)


def setup_test_environment():
    """Set up the test environment with optimal configuration."""
    env = detect_environment()

    print(f"Detected environment: {env.name}")

    # Set environment variables
    env_vars = env.get_env_vars()
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"  Set {key}={value}")

    # Return pytest args for use in scripts
    return env.get_pytest_args()


if __name__ == "__main__":
    # Print optimized command when run directly
    print("Optimized pytest command for current environment:")
    print(get_optimized_pytest_command())
