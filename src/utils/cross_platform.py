
"""Cross-platform utilities for handling platform-specific differences.

This module provides utilities for consistent behavior across Windows, macOS, and Linux
platforms, especially for file paths, environment variables, and system dependencies.
"""

import os
import platform
import sys
from pathlib import Path
from typing import Any


def is_windows() -> bool:
    """Check if running on Windows."""
    return platform.system().lower() == "windows"


def is_macos() -> bool:
    """Check if running on macOS."""
    return platform.system().lower() == "darwin"


def is_linux() -> bool:
    """Check if running on Linux."""
    return platform.system().lower() == "linux"


def is_ci_environment() -> bool:
    """Check if running in a CI environment."""
    ci_indicators = [
        "CI",
        "CONTINUOUS_INTEGRATION",
        "GITHUB_ACTIONS",
        "GITLAB_CI",
        "TRAVIS",
        "CIRCLECI",
        "JENKINS_URL",
        "BUILDKITE",
        "TF_BUILD",  # Azure DevOps
    ]
    return any(os.getenv(indicator) for indicator in ci_indicators)


def get_platform_cache_dir(app_name: str = "ai-docs-scraper") -> Path:
    """Get platform-appropriate cache directory.

    Args:
        app_name: Application name for cache directory

    Returns:
        Path: Platform-specific cache directory
    """
    if is_windows():
        cache_root = Path(os.getenv("LOCALAPPDATA", "~\\AppData\\Local")).expanduser()
    elif is_macos():
        cache_root = Path("~/Library/Caches").expanduser()
    else:  # Linux and other Unix-like
        cache_root = Path(os.getenv("XDG_CACHE_HOME", "~/.cache")).expanduser()

    return cache_root / app_name


def get_platform_config_dir(app_name: str = "ai-docs-scraper") -> Path:
    """Get platform-appropriate configuration directory.

    Args:
        app_name: Application name for config directory

    Returns:
        Path: Platform-specific config directory
    """
    if is_windows():
        config_root = Path(os.getenv("APPDATA", "~\\AppData\\Roaming")).expanduser()
    elif is_macos():
        config_root = Path("~/Library/Application Support").expanduser()
    else:  # Linux and other Unix-like
        config_root = Path(os.getenv("XDG_CONFIG_HOME", "~/.config")).expanduser()

    return config_root / app_name


def get_platform_data_dir(app_name: str = "ai-docs-scraper") -> Path:
    """Get platform-appropriate data directory.

    Args:
        app_name: Application name for data directory

    Returns:
        Path: Platform-specific data directory
    """
    if is_windows():
        data_root = Path(os.getenv("LOCALAPPDATA", "~\\AppData\\Local")).expanduser()
    elif is_macos():
        data_root = Path("~/Library/Application Support").expanduser()
    else:  # Linux and other Unix-like
        data_root = Path(os.getenv("XDG_DATA_HOME", "~/.local/share")).expanduser()

    return data_root / app_name


def normalize_path(path: str | Path) -> Path:
    """Normalize a path for the current platform.

    Args:
        path: Path to normalize

    Returns:
        Path: Normalized Path object
    """
    if isinstance(path, str):
        path = Path(path)

    # Convert to absolute path and resolve any symbolic links
    return path.expanduser().resolve()


def get_browser_executable_path(browser: str = "chromium") -> Path | None:
    """Get platform-specific browser executable path.

    Args:
        browser: Browser name (chromium, chrome, firefox)

    Returns:
        typing.Optional[Path]: Path to browser executable if found
    """
    browser_paths = {
        "chromium": {
            "windows": [
                "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
                "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
                "C:\\Users\\{username}\\AppData\\Local\\Google\\Chrome\\Application\\chrome.exe",
            ],
            "darwin": [
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                "/Applications/Chromium.app/Contents/MacOS/Chromium",
                "/usr/bin/chromium-browser",
                "/usr/bin/google-chrome",
            ],
            "linux": [
                "/usr/bin/chromium-browser",
                "/usr/bin/google-chrome",
                "/usr/bin/google-chrome-stable",
                "/snap/bin/chromium",
                "/usr/bin/chromium",
            ],
        }
    }

    system = platform.system().lower()
    if browser not in browser_paths or system not in browser_paths[browser]:
        return None

    username = os.getenv("USERNAME" if is_windows() else "USER", "")

    for path_template in browser_paths[browser][system]:
        path_str = path_template.format(username=username)
        path = Path(path_str)
        if path.exists():
            return path

    return None


def get_platform_temp_dir() -> Path:
    """Get platform-appropriate temporary directory."""
    if is_windows():
        temp_dir = Path(os.getenv("TEMP", os.getenv("TMP", "C:\\temp")))
    else:
        temp_dir = Path("/tmp")

    return temp_dir


def get_environment_variables(prefix: str = "AI_DOCS") -> dict[str, str]:
    """Get environment variables with platform-specific handling.

    Args:
        prefix: Environment variable prefix

    Returns:
        Dict[str, str]: Environment variables matching prefix
    """
    env_vars = {}
    prefix_upper = prefix.upper()

    for key, value in os.environ.items():
        if key.upper().startswith(prefix_upper):
            env_vars[key] = value

    return env_vars


def get_playwright_browser_path() -> str | None:
    """Get platform-specific Playwright browser path."""
    if is_windows():
        return os.getenv(
            "PLAYWRIGHT_BROWSERS_PATH",
            os.path.expanduser("~\\AppData\\Local\\ms-playwright"),
        )
    elif is_macos():
        return os.getenv(
            "PLAYWRIGHT_BROWSERS_PATH",
            os.path.expanduser("~/Library/Caches/ms-playwright"),
        )
    else:  # Linux
        return os.getenv(
            "PLAYWRIGHT_BROWSERS_PATH", os.path.expanduser("~/.cache/ms-playwright")
        )


def get_platform_python_executable() -> Path:
    """Get current Python executable path."""
    return Path(sys.executable)


def get_platform_shell_command() -> list[str]:
    """Get platform-appropriate shell command for subprocess."""
    if is_windows():
        return ["cmd", "/c"]
    else:
        return ["/bin/bash", "-c"]


def set_platform_environment_defaults() -> dict[str, str]:
    """Set platform-specific environment variable defaults.

    Returns:
        Dict[str, str]: Environment variables to set
    """
    env_defaults = {}

    # Common settings
    env_defaults["PYTHONIOENCODING"] = "utf-8"
    env_defaults["PYTHONUNBUFFERED"] = "1"

    if is_windows():
        env_defaults["PYTHONUTF8"] = "1"
        env_defaults["PLAYWRIGHT_BROWSERS_PATH"] = get_playwright_browser_path()
    elif is_ci_environment():
        env_defaults["PLAYWRIGHT_BROWSERS_PATH"] = "0"
        env_defaults["CRAWL4AI_HEADLESS"] = "true"

    return env_defaults


def get_file_permissions() -> int:
    """Get appropriate file permissions for the platform."""
    if is_windows():
        # Windows doesn't use Unix-style permissions
        return 0o777
    else:
        # Unix-like systems: read/write for owner, read for group/others
        return 0o644


def get_directory_permissions() -> int:
    """Get appropriate directory permissions for the platform."""
    if is_windows():
        # Windows doesn't use Unix-style permissions
        return 0o777
    else:
        # Unix-like systems: full access for owner, read/execute for group/others
        return 0o755


def create_directory_with_permissions(path: Path) -> None:
    """Create directory with platform-appropriate permissions.

    Args:
        path: Directory path to create
    """
    path.mkdir(parents=True, exist_ok=True)

    if not is_windows():
        # Set permissions only on Unix-like systems
        path.chmod(get_directory_permissions())


def write_file_with_permissions(path: Path, content: str) -> None:
    """Write file with platform-appropriate permissions.

    Args:
        path: File path to write
        content: Content to write
    """
    path.write_text(content, encoding="utf-8")

    if not is_windows():
        # Set permissions only on Unix-like systems
        path.chmod(get_file_permissions())


def get_process_info() -> dict[str, Any]:
    """Get platform-specific process information.

    Returns:
        Dict[str, Any]: Process information including platform details
    """
    return {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "is_windows": is_windows(),
        "is_macos": is_macos(),
        "is_linux": is_linux(),
        "is_ci": is_ci_environment(),
        "executable": str(get_platform_python_executable()),
        "cache_dir": str(get_platform_cache_dir()),
        "config_dir": str(get_platform_config_dir()),
        "data_dir": str(get_platform_data_dir()),
    }
