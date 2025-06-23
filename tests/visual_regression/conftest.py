"""Visual regression testing fixtures and configuration.

This module provides pytest fixtures for comprehensive visual regression testing including
screenshot capture, baseline management, visual comparison, responsive testing,
and UI component validation.
"""

import base64
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import AsyncMock, MagicMock

import pytest


@dataclass
class VisualTestConfig:
    """Configuration for visual regression testing."""
    
    test_name: str
    url: str
    viewport_width: int = 1280
    viewport_height: int = 720
    wait_for_selector: Optional[str] = None
    wait_time: float = 1.0
    hide_selectors: List[str] = field(default_factory=list)
    mask_selectors: List[str] = field(default_factory=list)
    full_page: bool = False
    threshold: float = 0.01  # Difference threshold (0-1)
    ignore_antialiasing: bool = True
    ignore_colors: List[str] = field(default_factory=list)


@dataclass
class Screenshot:
    """Screenshot data and metadata."""
    
    name: str
    data: bytes
    width: int
    height: int
    format: str = "png"
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def base64_data(self) -> str:
        """Get base64 encoded screenshot data."""
        return base64.b64encode(self.data).decode('utf-8')
    
    @property
    def hash(self) -> str:
        """Get SHA-256 hash of screenshot data."""
        return hashlib.sha256(self.data).hexdigest()


@dataclass
class VisualComparisonResult:
    """Result of visual comparison."""
    
    test_name: str
    baseline_hash: str
    current_hash: str
    difference_percentage: float
    threshold: float
    passed: bool
    diff_image_data: Optional[bytes] = None
    regions_changed: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResponsiveTestConfig:
    """Configuration for responsive visual testing."""
    
    viewports: List[Dict[str, int]] = field(default_factory=lambda: [
        {"width": 320, "height": 568, "name": "mobile"},
        {"width": 768, "height": 1024, "name": "tablet"},
        {"width": 1280, "height": 720, "name": "desktop"},
        {"width": 1920, "height": 1080, "name": "desktop_hd"},
    ])
    orientations: List[str] = field(default_factory=lambda: ["portrait", "landscape"])
    test_interactions: bool = True
    test_hover_states: bool = True


@pytest.fixture(scope="session")
def visual_regression_config():
    """Provide visual regression testing configuration."""
    return {
        "screenshots": {
            "format": "png",
            "quality": 90,
            "baseline_dir": "tests/visual_regression/baseline",
            "current_dir": "tests/visual_regression/screenshots/current",
            "diff_dir": "tests/visual_regression/screenshots/diff",
            "cleanup_after_test": False,
            "compression": True,
        },
        "comparison": {
            "default_threshold": 0.01,
            "pixel_match_threshold": 0.1,
            "ignore_antialiasing": True,
            "ignore_colors": ["#ff0000", "#00ff00"],  # Red and green for debugging
            "diff_color": "#ff00ff",  # Magenta for differences
            "enable_subpixel_matching": True,
        },
        "responsive": {
            "default_viewports": [
                {"width": 320, "height": 568, "name": "mobile_portrait"},
                {"width": 568, "height": 320, "name": "mobile_landscape"},
                {"width": 768, "height": 1024, "name": "tablet_portrait"},
                {"width": 1024, "height": 768, "name": "tablet_landscape"},
                {"width": 1280, "height": 720, "name": "desktop"},
                {"width": 1920, "height": 1080, "name": "desktop_hd"},
            ],
            "interaction_delays": {
                "hover": 0.5,
                "click": 0.3,
                "focus": 0.2,
            },
        },
        "browser": {
            "headless": True,
            "disable_web_security": False,
            "ignore_certificate_errors": True,
            "timeout": 30000,
            "user_agent": "Mozilla/5.0 (compatible; VisualRegressionBot/1.0)",
        },
        "performance": {
            "parallel_screenshots": True,
            "max_concurrent": 4,
            "cache_screenshots": True,
            "optimize_images": True,
        },
    }


@pytest.fixture
def screenshot_manager():
    """Screenshot capture and management utilities."""
    
    class ScreenshotManager:
        def __init__(self):
            self.screenshots = {}
            self.baseline_dir = Path("tests/visual_regression/baseline")
            self.current_dir = Path("tests/visual_regression/screenshots/current")
            self.diff_dir = Path("tests/visual_regression/screenshots/diff")
            
            # Ensure directories exist
            for directory in [self.baseline_dir, self.current_dir, self.diff_dir]:
                directory.mkdir(parents=True, exist_ok=True)
        
        async def capture_screenshot(self, page, config: VisualTestConfig) -> Screenshot:
            """Capture screenshot with configuration."""
            # Set viewport
            await page.set_viewport_size(config.viewport_width, config.viewport_height)
            
            # Navigate to URL
            await page.goto(config.url)
            
            # Wait for specific selector if provided
            if config.wait_for_selector:
                await page.wait_for_selector(config.wait_for_selector)
            
            # Wait for specified time
            if config.wait_time > 0:
                await page.wait_for_timeout(int(config.wait_time * 1000))
            
            # Hide elements if specified
            for selector in config.hide_selectors:
                await page.evaluate(f"document.querySelectorAll('{selector}').forEach(el => el.style.visibility = 'hidden')")
            
            # Mask elements if specified
            for selector in config.mask_selectors:
                await page.evaluate(f"document.querySelectorAll('{selector}').forEach(el => el.style.background = '#000000')")
            
            # Capture screenshot
            screenshot_options = {
                "full_page": config.full_page,
                "type": "png",
            }
            
            screenshot_data = await page.screenshot(**screenshot_options)
            
            # Get viewport size for metadata
            viewport = await page.evaluate("() => ({ width: window.innerWidth, height: window.innerHeight })")
            
            screenshot = Screenshot(
                name=config.test_name,
                data=screenshot_data,
                width=viewport["width"],
                height=viewport["height"],
                metadata={
                    "url": config.url,
                    "viewport": {"width": config.viewport_width, "height": config.viewport_height},
                    "full_page": config.full_page,
                    "user_agent": await page.evaluate("navigator.userAgent"),
                }
            )
            
            self.screenshots[config.test_name] = screenshot
            return screenshot
        
        def save_screenshot(self, screenshot: Screenshot, directory: Path, filename: Optional[str] = None) -> Path:
            """Save screenshot to disk."""
            if not filename:
                filename = f"{screenshot.name}.png"
            
            file_path = directory / filename
            file_path.write_bytes(screenshot.data)
            
            # Save metadata
            metadata_path = directory / f"{screenshot.name}_metadata.json"
            metadata = {
                "name": screenshot.name,
                "timestamp": screenshot.timestamp.isoformat(),
                "width": screenshot.width,
                "height": screenshot.height,
                "format": screenshot.format,
                "hash": screenshot.hash,
                "metadata": screenshot.metadata,
            }
            metadata_path.write_text(json.dumps(metadata, indent=2))
            
            return file_path
        
        def load_screenshot(self, name: str, directory: Path) -> Optional[Screenshot]:
            """Load screenshot from disk."""
            file_path = directory / f"{name}.png"
            metadata_path = directory / f"{name}_metadata.json"
            
            if not file_path.exists():
                return None
            
            screenshot_data = file_path.read_bytes()
            
            # Load metadata if available
            metadata = {}
            if metadata_path.exists():
                try:
                    metadata = json.loads(metadata_path.read_text())
                except json.JSONDecodeError:
                    pass
            
            return Screenshot(
                name=name,
                data=screenshot_data,
                width=metadata.get("width", 0),
                height=metadata.get("height", 0),
                format=metadata.get("format", "png"),
                timestamp=datetime.fromisoformat(metadata["timestamp"]) if "timestamp" in metadata else datetime.now(),
                metadata=metadata.get("metadata", {}),
            )
        
        def get_baseline_screenshot(self, test_name: str) -> Optional[Screenshot]:
            """Get baseline screenshot for comparison."""
            return self.load_screenshot(test_name, self.baseline_dir)
        
        def save_as_baseline(self, screenshot: Screenshot) -> Path:
            """Save screenshot as new baseline."""
            return self.save_screenshot(screenshot, self.baseline_dir)
        
        def cleanup_current_screenshots(self):
            """Clean up current screenshots directory."""
            for file_path in self.current_dir.glob("*"):
                if file_path.is_file():
                    file_path.unlink()
    
    return ScreenshotManager()


@pytest.fixture
def visual_comparator():
    """Visual comparison utilities."""
    
    class VisualComparator:
        def __init__(self):
            self.comparison_results = []
        
        def compare_screenshots(self, baseline: Screenshot, current: Screenshot, config: VisualTestConfig) -> VisualComparisonResult:
            """Compare two screenshots."""
            # Simple hash-based comparison for mock implementation
            # In real implementation, you would use image comparison libraries
            
            baseline_hash = baseline.hash
            current_hash = current.hash
            
            # Calculate mock difference percentage
            if baseline_hash == current_hash:
                difference_percentage = 0.0
            else:
                # Mock calculation based on hash differences
                hash_diff = sum(c1 != c2 for c1, c2 in zip(baseline_hash, current_hash))
                difference_percentage = min(hash_diff / len(baseline_hash), 1.0)
            
            passed = difference_percentage <= config.threshold
            
            # Generate mock diff regions if there are differences
            regions_changed = []
            if difference_percentage > 0:
                regions_changed = [
                    {
                        "x": 100,
                        "y": 50,
                        "width": 200,
                        "height": 150,
                        "confidence": 0.8,
                    },
                    {
                        "x": 400,
                        "y": 300,
                        "width": 100,
                        "height": 75,
                        "confidence": 0.6,
                    },
                ]
            
            result = VisualComparisonResult(
                test_name=config.test_name,
                baseline_hash=baseline_hash,
                current_hash=current_hash,
                difference_percentage=difference_percentage,
                threshold=config.threshold,
                passed=passed,
                regions_changed=regions_changed,
                metadata={
                    "baseline_size": len(baseline.data),
                    "current_size": len(current.data),
                    "comparison_timestamp": datetime.now().isoformat(),
                    "comparison_algorithm": "hash_based_mock",
                }
            )
            
            self.comparison_results.append(result)
            return result
        
        def generate_diff_image(self, baseline: Screenshot, current: Screenshot) -> bytes:
            """Generate difference image (mock implementation)."""
            # In real implementation, this would create an actual diff image
            # For mock, return a placeholder diff image
            return b"MOCK_DIFF_IMAGE_DATA"
        
        def analyze_differences(self, result: VisualComparisonResult) -> Dict[str, Any]:
            """Analyze differences in detail."""
            analysis = {
                "total_changed_pixels": int(result.difference_percentage * 1000000),  # Mock calculation
                "change_intensity": "high" if result.difference_percentage > 0.1 else "medium" if result.difference_percentage > 0.01 else "low",
                "affected_regions": len(result.regions_changed),
                "largest_change_area": max((r["width"] * r["height"] for r in result.regions_changed), default=0),
                "change_distribution": {
                    "top_half": 0.6,
                    "bottom_half": 0.4,
                    "left_half": 0.3,
                    "right_half": 0.7,
                },
                "color_changes": {
                    "hue_shift": 0.02,
                    "saturation_change": 0.01,
                    "brightness_change": 0.03,
                },
            }
            
            return analysis
    
    return VisualComparator()


@pytest.fixture
def responsive_tester():
    """Responsive design testing utilities."""
    
    class ResponsiveTester:
        def __init__(self):
            self.test_results = []
        
        async def test_responsive_design(self, page, url: str, config: ResponsiveTestConfig) -> List[Dict[str, Any]]:
            """Test responsive design across multiple viewports."""
            results = []
            
            for viewport in config.viewports:
                viewport_result = {
                    "viewport": viewport,
                    "screenshots": {},
                    "layout_issues": [],
                    "performance_metrics": {},
                }
                
                # Test both orientations if specified
                orientations = config.orientations if len(config.orientations) > 1 else ["portrait"]
                
                for orientation in orientations:
                    if orientation == "landscape":
                        width, height = viewport["height"], viewport["width"]
                    else:
                        width, height = viewport["width"], viewport["height"]
                    
                    # Set viewport
                    await page.set_viewport_size(width, height)
                    await page.goto(url)
                    await page.wait_for_timeout(1000)  # Wait for layout
                    
                    # Capture screenshot
                    screenshot_name = f"{viewport['name']}_{orientation}"
                    screenshot_data = await page.screenshot(full_page=True)
                    
                    viewport_result["screenshots"][orientation] = {
                        "name": screenshot_name,
                        "data": screenshot_data,
                        "width": width,
                        "height": height,
                    }
                    
                    # Check for layout issues
                    layout_issues = await self._check_layout_issues(page, width, height)
                    viewport_result["layout_issues"].extend(layout_issues)
                    
                    # Test interactions if enabled
                    if config.test_interactions:
                        interaction_results = await self._test_interactions(page)
                        viewport_result[f"interactions_{orientation}"] = interaction_results
                    
                    # Test hover states if enabled
                    if config.test_hover_states:
                        hover_results = await self._test_hover_states(page)
                        viewport_result[f"hover_states_{orientation}"] = hover_results
                
                results.append(viewport_result)
            
            self.test_results.extend(results)
            return results
        
        async def _check_layout_issues(self, page, width: int, height: int) -> List[Dict[str, Any]]:
            """Check for common responsive layout issues."""
            issues = []
            
            # Check for horizontal overflow
            overflow_check = await page.evaluate("""
                () => {
                    const elements = document.querySelectorAll('*');
                    const overflowing = [];
                    for (let el of elements) {
                        const rect = el.getBoundingClientRect();
                        if (rect.right > window.innerWidth) {
                            overflowing.push({
                                tag: el.tagName,
                                class: el.className,
                                id: el.id,
                                overflowAmount: rect.right - window.innerWidth
                            });
                        }
                    }
                    return overflowing;
                }
            """)
            
            if overflow_check:
                issues.append({
                    "type": "horizontal_overflow",
                    "severity": "medium",
                    "elements": overflow_check[:5],  # Limit to first 5
                    "description": "Elements extending beyond viewport width"
                })
            
            # Check for tiny text
            tiny_text_check = await page.evaluate("""
                () => {
                    const elements = document.querySelectorAll('*');
                    const tinyText = [];
                    for (let el of elements) {
                        const style = window.getComputedStyle(el);
                        const fontSize = parseFloat(style.fontSize);
                        if (fontSize < 12 && el.textContent.trim()) {
                            tinyText.push({
                                tag: el.tagName,
                                fontSize: fontSize,
                                text: el.textContent.substring(0, 50)
                            });
                        }
                    }
                    return tinyText;
                }
            """)
            
            if tiny_text_check:
                issues.append({
                    "type": "tiny_text",
                    "severity": "low",
                    "elements": tiny_text_check[:5],
                    "description": "Text smaller than 12px may be hard to read"
                })
            
            # Check for overlapping elements
            overlap_check = await page.evaluate("""
                () => {
                    // Simplified overlap detection
                    const elements = Array.from(document.querySelectorAll('div, span, p, a, button'));
                    const overlaps = [];
                    for (let i = 0; i < elements.length - 1; i++) {
                        const rect1 = elements[i].getBoundingClientRect();
                        const rect2 = elements[i + 1].getBoundingClientRect();
                        
                        if (rect1.left < rect2.right && rect2.left < rect1.right &&
                            rect1.top < rect2.bottom && rect2.top < rect1.bottom) {
                            overlaps.push({
                                element1: elements[i].tagName + (elements[i].className ? '.' + elements[i].className.split(' ')[0] : ''),
                                element2: elements[i + 1].tagName + (elements[i + 1].className ? '.' + elements[i + 1].className.split(' ')[0] : '')
                            });
                        }
                    }
                    return overlaps.slice(0, 3); // Limit results
                }
            """)
            
            if overlap_check:
                issues.append({
                    "type": "overlapping_elements",
                    "severity": "high",
                    "elements": overlap_check,
                    "description": "Elements may be overlapping"
                })
            
            return issues
        
        async def _test_interactions(self, page) -> Dict[str, Any]:
            """Test basic interactions on responsive design."""
            results = {
                "clickable_elements": 0,
                "touch_targets_too_small": 0,
                "successful_interactions": 0,
                "failed_interactions": 0,
            }
            
            # Find clickable elements
            clickable_elements = await page.evaluate("""
                () => {
                    const clickable = document.querySelectorAll('button, a, input[type="submit"], [onclick], [role="button"]');
                    return Array.from(clickable).map(el => {
                        const rect = el.getBoundingClientRect();
                        return {
                            tag: el.tagName,
                            width: rect.width,
                            height: rect.height,
                            x: rect.x,
                            y: rect.y
                        };
                    });
                }
            """)
            
            results["clickable_elements"] = len(clickable_elements)
            
            # Check touch target sizes (should be at least 44x44px)
            small_targets = [el for el in clickable_elements if el["width"] < 44 or el["height"] < 44]
            results["touch_targets_too_small"] = len(small_targets)
            
            return results
        
        async def _test_hover_states(self, page) -> Dict[str, Any]:
            """Test hover states on interactive elements."""
            results = {
                "elements_with_hover": 0,
                "hover_effects_detected": 0,
            }
            
            # Find elements with hover effects
            hover_elements = await page.evaluate("""
                () => {
                    const styles = Array.from(document.styleSheets);
                    let hoverRules = 0;
                    
                    try {
                        for (let sheet of styles) {
                            for (let rule of sheet.cssRules || []) {
                                if (rule.selectorText && rule.selectorText.includes(':hover')) {
                                    hoverRules++;
                                }
                            }
                        }
                    } catch (e) {
                        // Cross-origin stylesheets may not be accessible
                    }
                    
                    return hoverRules;
                }
            """)
            
            results["hover_effects_detected"] = hover_elements
            
            return results
    
    return ResponsiveTester()


@pytest.fixture
def mock_browser_page():
    """Mock browser page for testing without real browser."""
    page = AsyncMock()
    
    # Mock basic page methods
    page.goto = AsyncMock()
    page.screenshot = AsyncMock(return_value=b"MOCK_SCREENSHOT_DATA")
    page.set_viewport_size = AsyncMock()
    page.wait_for_selector = AsyncMock()
    page.wait_for_timeout = AsyncMock()
    page.evaluate = AsyncMock(return_value={"width": 1280, "height": 720})
    
    return page


@pytest.fixture
def visual_regression_test_data():
    """Provide test data for visual regression testing."""
    return {
        "sample_configs": [
            VisualTestConfig(
                test_name="homepage",
                url="http://localhost:3000",
                viewport_width=1280,
                viewport_height=720,
                threshold=0.01,
            ),
            VisualTestConfig(
                test_name="mobile_homepage",
                url="http://localhost:3000",
                viewport_width=375,
                viewport_height=667,
                threshold=0.02,
            ),
            VisualTestConfig(
                test_name="search_page",
                url="http://localhost:3000/search",
                wait_for_selector=".search-results",
                hide_selectors=[".ads", ".dynamic-content"],
                mask_selectors=[".user-avatar", ".timestamp"],
                threshold=0.015,
            ),
        ],
        "responsive_config": ResponsiveTestConfig(
            viewports=[
                {"width": 320, "height": 568, "name": "mobile"},
                {"width": 768, "height": 1024, "name": "tablet"},
                {"width": 1280, "height": 720, "name": "desktop"},
            ],
            test_interactions=True,
            test_hover_states=True,
        ),
    }


# Pytest markers for visual regression test categorization
def pytest_configure(config):
    """Configure visual regression testing markers."""
    config.addinivalue_line(
        "markers", "visual_regression: mark test as visual regression test"
    )
    config.addinivalue_line(
        "markers", "visual: mark test as visual test"
    )
    config.addinivalue_line(
        "markers", "responsive: mark test as responsive design test"
    )
    config.addinivalue_line(
        "markers", "screenshot: mark test as screenshot comparison test"
    )
    config.addinivalue_line(
        "markers", "ui_components: mark test as UI component visual test"
    )
    config.addinivalue_line(
        "markers", "cross_browser: mark test as cross-browser visual test"
    )