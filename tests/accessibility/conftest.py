"""Accessibility testing fixtures and configuration.

This module provides pytest fixtures for comprehensive accessibility testing including
WCAG compliance, screen reader compatibility, keyboard navigation, color contrast,
and ARIA attribute validation.
"""

import re
from typing import Any
from unittest.mock import MagicMock

import pytest


@pytest.fixture(scope="session")
def accessibility_test_config():
    """Provide accessibility testing configuration."""
    return {
        "wcag": {
            "version": "2.1",
            "conformance_level": "AA",  # A, AA, or AAA
            "guidelines": {
                "perceivable": True,
                "operable": True,
                "understandable": True,
                "robust": True,
            },
        },
        "color_contrast": {
            "normal_text_ratio": 4.5,  # WCAG AA requirement
            "large_text_ratio": 3.0,  # WCAG AA requirement for large text
            "enhanced_ratio": 7.0,  # WCAG AAA requirement
        },
        "keyboard_navigation": {
            "tab_order_validation": True,
            "focus_indicators": True,
            "skip_links": True,
            "keyboard_shortcuts": True,
        },
        "screen_reader": {
            "aria_support": True,
            "semantic_html": True,
            "alt_text_validation": True,
            "heading_structure": True,
        },
        "testing_tools": {
            "axe_core": True,
            "lighthouse": True,
            "wave": True,
            "pa11y": True,
        },
    }


@pytest.fixture
def wcag_validator():
    """WCAG compliance validation utilities."""

    class WCAGValidator:
        def __init__(self):
            self.guidelines = {
                "1.1.1": "Non-text Content",
                "1.2.1": "Audio-only and Video-only (Prerecorded)",
                "1.2.2": "Captions (Prerecorded)",
                "1.2.3": "Audio Description or Media Alternative (Prerecorded)",
                "1.3.1": "Info and Relationships",
                "1.3.2": "Meaningful Sequence",
                "1.3.3": "Sensory Characteristics",
                "1.4.1": "Use of Color",
                "1.4.2": "Audio Control",
                "1.4.3": "Contrast (Minimum)",
                "1.4.4": "Resize text",
                "1.4.5": "Images of Text",
                "2.1.1": "Keyboard",
                "2.1.2": "No Keyboard Trap",
                "2.1.4": "Character Key Shortcuts",
                "2.2.1": "Timing Adjustable",
                "2.2.2": "Pause, Stop, Hide",
                "2.3.1": "Three Flashes or Below Threshold",
                "2.4.1": "Bypass Blocks",
                "2.4.2": "Page Titled",
                "2.4.3": "Focus Order",
                "2.4.4": "Link Purpose (In Context)",
                "2.4.5": "Multiple Ways",
                "2.4.6": "Headings and Labels",
                "2.4.7": "Focus Visible",
                "3.1.1": "Language of Page",
                "3.1.2": "Language of Parts",
                "3.2.1": "On Focus",
                "3.2.2": "On Input",
                "3.2.3": "Consistent Navigation",
                "3.2.4": "Consistent Identification",
                "3.3.1": "Error Identification",
                "3.3.2": "Labels or Instructions",
                "3.3.3": "Error Suggestion",
                "3.3.4": "Error Prevention (Legal, Financial, Data)",
                "4.1.1": "Parsing",
                "4.1.2": "Name, Role, Value",
                "4.1.3": "Status Messages",
            }

        def validate_html_structure(self, html_content: str) -> dict[str, Any]:
            """Validate HTML structure for accessibility."""
            issues = []

            # Check for missing lang attribute
            if "lang=" not in html_content:
                issues.append(
                    {
                        "guideline": "3.1.1",
                        "level": "AA",
                        "issue": "Missing lang attribute on html element",
                        "severity": "error",
                    }
                )

            # Check for missing alt attributes on images
            img_pattern = r"<img(?![^>]*alt=)[^>]*>"
            missing_alt = re.findall(img_pattern, html_content, re.IGNORECASE)
            if missing_alt:
                issues.append(
                    {
                        "guideline": "1.1.1",
                        "level": "A",
                        "issue": f"Found {len(missing_alt)} images without alt attributes",
                        "severity": "error",
                    }
                )

            # Check for proper heading hierarchy
            headings = re.findall(r"<h([1-6])[^>]*>", html_content, re.IGNORECASE)
            if headings:
                heading_levels = [int(h) for h in headings]
                if heading_levels and heading_levels[0] != 1:
                    issues.append(
                        {
                            "guideline": "1.3.1",
                            "level": "A",
                            "issue": "Page should start with h1 heading",
                            "severity": "warning",
                        }
                    )

                # Check for skipped heading levels
                issues.extend(
                    [
                        {
                            "guideline": "1.3.1",
                            "level": "A",
                            "issue": f"Skipped heading level: h{heading_levels[i - 1]} to h{heading_levels[i]}",
                            "severity": "warning",
                        }
                        for i in range(1, len(heading_levels))
                        if heading_levels[i] > heading_levels[i - 1] + 1
                    ]
                )

            # Check for missing form labels
            input_pattern = r'<input(?![^>]*(?:type=["\'](?:hidden|submit|button)["\']|id=["\']([^"\']*)["\']))[^>]*>'
            inputs_without_labels = re.findall(
                input_pattern, html_content, re.IGNORECASE
            )
            if inputs_without_labels:
                issues.append(
                    {
                        "guideline": "3.3.2",
                        "level": "A",
                        "issue": "Form inputs without proper labels",
                        "severity": "error",
                    }
                )

            return {
                "compliant": len([i for i in issues if i["severity"] == "error"]) == 0,
                "issues": issues,
                "_total_issues": len(issues),
                "errors": len([i for i in issues if i["severity"] == "error"]),
                "warnings": len([i for i in issues if i["severity"] == "warning"]),
            }

        def validate_aria_attributes(self, html_content: str) -> dict[str, Any]:
            """Validate ARIA attributes."""
            issues = []

            # Check for required aria-label or aria-labelledby
            interactive_elements = [
                r"<button[^>]*>",
                r'<input[^>]*type=["\']button["\'][^>]*>',
                r"<a[^>]*href=[^>]*>",
            ]

            for pattern in interactive_elements:
                elements = re.findall(pattern, html_content, re.IGNORECASE)
                for element in elements:
                    if "aria-label" not in element and "aria-labelledby" not in element:
                        # Check if element has visible text content
                        text_match = re.search(r">([^<]+)</", element + "dummy>")
                        if not text_match or not text_match.group(1).strip():
                            issues.append(
                                {
                                    "guideline": "4.1.2",
                                    "level": "A",
                                    "issue": "Interactive element without accessible name",
                                    "element": element[:100] + "..."
                                    if len(element) > 100
                                    else element,
                                    "severity": "error",
                                }
                            )

            # Check for invalid ARIA attributes
            aria_pattern = r"aria-([a-z-]+)="
            aria_attributes = re.findall(aria_pattern, html_content, re.IGNORECASE)

            valid_aria_attributes = {
                "label",
                "labelledby",
                "describedby",
                "hidden",
                "expanded",
                "selected",
                "checked",
                "disabled",
                "required",
                "invalid",
                "live",
                "atomic",
                "relevant",
                "busy",
                "controls",
                "owns",
                "flowto",
                "activedescendant",
                "level",
                "setsize",
                "posinset",
                "orientation",
                "valuemin",
                "valuemax",
                "valuenow",
                "valuetext",
                "sort",
                "readonly",
                "multiline",
                "multiselectable",
                "autocomplete",
                "haspopup",
                "pressed",
                "dropeffect",
                "grabbed",
                "current",
                "details",
                "errormessage",
                "keyshortcuts",
                "modal",
                "placeholder",
                "roledescription",
                "colcount",
                "colindex",
                "colspan",
                "rowcount",
                "rowindex",
                "rowspan",
            }

            issues.extend(
                [
                    {
                        "guideline": "4.1.2",
                        "level": "A",
                        "issue": f"Invalid ARIA attribute: aria-{attr}",
                        "severity": "error",
                    }
                    for attr in aria_attributes
                    if attr.lower() not in valid_aria_attributes
                ]
            )

            return {
                "compliant": len([i for i in issues if i["severity"] == "error"]) == 0,
                "issues": issues,
                "_total_attributes": len(aria_attributes),
                "unique_attributes": len(set(aria_attributes)),
            }

    return WCAGValidator()


@pytest.fixture
def color_contrast_analyzer():
    """Color contrast analysis utilities."""

    class ColorContrastAnalyzer:
        def __init__(self):
            self.wcag_aa_normal = 4.5
            self.wcag_aa_large = 3.0
            self.wcag_aaa_normal = 7.0
            self.wcag_aaa_large = 4.5

        def hex_to_rgb(self, hex_color: str) -> tuple[int, int, int]:
            """Convert hex color to RGB tuple."""
            hex_color = hex_color.lstrip("#")
            if len(hex_color) == 3:
                hex_color = "".join([c * 2 for c in hex_color])
            return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

        def get_relative_luminance(self, rgb: tuple[int, int, int]) -> float:
            """Calculate relative luminance of RGB color."""

            def gamma_correct(value):
                value = value / 255.0
                if value <= 0.03928:
                    return value / 12.92
                return pow((value + 0.055) / 1.055, 2.4)

            r, g, b = rgb
            r_linear = gamma_correct(r)
            g_linear = gamma_correct(g)
            b_linear = gamma_correct(b)

            return 0.2126 * r_linear + 0.7152 * g_linear + 0.0722 * b_linear

        def calculate_contrast_ratio(self, color1: str, color2: str) -> float:
            """Calculate contrast ratio between two colors."""
            rgb1 = self.hex_to_rgb(color1)
            rgb2 = self.hex_to_rgb(color2)

            lum1 = self.get_relative_luminance(rgb1)
            lum2 = self.get_relative_luminance(rgb2)

            # Ensure lighter color is in numerator
            if lum1 > lum2:
                return (lum1 + 0.05) / (lum2 + 0.05)
            return (lum2 + 0.05) / (lum1 + 0.05)

        def check_contrast_compliance(
            self, foreground: str, background: str, text_size: str = "normal"
        ) -> dict[str, Any]:
            """Check if color combination meets WCAG contrast requirements."""
            ratio = self.calculate_contrast_ratio(foreground, background)

            # Determine thresholds based on text size
            if text_size == "large":
                aa_threshold = self.wcag_aa_large
                aaa_threshold = self.wcag_aaa_large
            else:
                aa_threshold = self.wcag_aa_normal
                aaa_threshold = self.wcag_aaa_normal

            return {
                "contrast_ratio": round(ratio, 2),
                "foreground": foreground,
                "background": background,
                "text_size": text_size,
                "wcag_aa": ratio >= aa_threshold,
                "wcag_aaa": ratio >= aaa_threshold,
                "aa_threshold": aa_threshold,
                "aaa_threshold": aaa_threshold,
                "meets_minimum": ratio >= aa_threshold,
            }

        def generate_accessible_color_suggestions(
            self, base_color: str, target_background: str
        ) -> list[dict[str, Any]]:
            """Generate accessible color suggestions."""
            suggestions = []
            base_rgb = self.hex_to_rgb(base_color)

            # Try variations of the base color
            for adjustment in [-60, -40, -20, 20, 40, 60, 80, 100]:
                new_rgb = tuple(
                    max(0, min(255, component + adjustment)) for component in base_rgb
                )
                new_hex = "#{:02x}{:02x}{:02x}".format(*new_rgb)

                compliance = self.check_contrast_compliance(new_hex, target_background)
                if compliance["wcag_aa"]:
                    suggestions.append(
                        {
                            "color": new_hex,
                            "contrast_ratio": compliance["contrast_ratio"],
                            "wcag_aaa": compliance["wcag_aaa"],
                        }
                    )

            # Sort by contrast ratio (highest first)
            return sorted(suggestions, key=lambda x: x["contrast_ratio"], reverse=True)

    return ColorContrastAnalyzer()


@pytest.fixture
def keyboard_navigation_tester():
    """Keyboard navigation testing utilities."""

    class KeyboardNavigationTester:
        def __init__(self):
            self.focusable_selectors = [
                "a[href]",
                "button:not([disabled])",
                'input:not([disabled]):not([type="hidden"])',
                "select:not([disabled])",
                "textarea:not([disabled])",
                '[tabindex]:not([tabindex="-1"])',
                "details",
                "summary",
            ]

        def validate_tab_order(self, html_content: str) -> dict[str, Any]:
            """Validate tab order in HTML content."""
            issues = []

            # Extract elements with tabindex
            tabindex_pattern = r'<[^>]*tabindex=["\']([^"\']*)["\'][^>]*>'
            tabindex_matches = re.findall(tabindex_pattern, html_content, re.IGNORECASE)

            tabindex_values = []
            for match in tabindex_matches:
                try:
                    tabindex_values.append(int(match))
                except ValueError:
                    issues.append(
                        {
                            "issue": f"Invalid tabindex value: {match}",
                            "severity": "error",
                            "guideline": "2.4.3",
                        }
                    )

            # Check for problematic tabindex values
            positive_tabindex = [t for t in tabindex_values if t > 0]
            if positive_tabindex:
                issues.append(
                    {
                        "issue": f"Positive tabindex values found: {positive_tabindex}",
                        "severity": "warning",
                        "guideline": "2.4.3",
                        "recommendation": "Use tabindex='0' or remove tabindex attribute",
                    }
                )

            # Check for tabindex="-1" on interactive elements
            negative_tabindex_pattern = (
                r'<(button|input|select|textarea|a)[^>]*tabindex=["\'](-1)["\'][^>]*>'
            )
            negative_matches = re.findall(
                negative_tabindex_pattern, html_content, re.IGNORECASE
            )
            if negative_matches:
                issues.append(
                    {
                        "issue": f"Interactive elements with tabindex='-1': {len(negative_matches)}",
                        "severity": "warning",
                        "guideline": "2.1.1",
                    }
                )

            return {
                "compliant": len([i for i in issues if i["severity"] == "error"]) == 0,
                "issues": issues,
                "_total_tabindex_elements": len(tabindex_matches),
                "positive_tabindex_count": len(positive_tabindex),
            }

        def check_focus_indicators(self, css_content: str = "") -> dict[str, Any]:
            """Check for focus indicators in CSS."""
            issues = []

            if not css_content:
                issues.append(
                    {
                        "issue": "No CSS content provided for focus indicator check",
                        "severity": "warning",
                        "guideline": "2.4.7",
                    }
                )
                return {"compliant": False, "issues": issues}

            # Check for :focus pseudo-class usage
            focus_pattern = r":focus\s*\{"
            focus_matches = re.findall(focus_pattern, css_content, re.IGNORECASE)

            if not focus_matches:
                issues.append(
                    {
                        "issue": "No :focus styles found in CSS",
                        "severity": "error",
                        "guideline": "2.4.7",
                        "recommendation": "Add visible focus indicators for all interactive elements",
                    }
                )

            # Check for outline removal without replacement
            outline_none_pattern = r"outline\s*:\s*none"
            outline_none_matches = re.findall(
                outline_none_pattern, css_content, re.IGNORECASE
            )

            if outline_none_matches:
                # Check if there are alternative focus indicators
                focus_alternatives = [
                    r"border.*:focus",
                    r"box-shadow.*:focus",
                    r"background.*:focus",
                ]

                has_alternatives = any(
                    re.search(pattern, css_content, re.IGNORECASE)
                    for pattern in focus_alternatives
                )

                if not has_alternatives:
                    issues.append(
                        {
                            "issue": "outline:none used without alternative focus indicators",
                            "severity": "error",
                            "guideline": "2.4.7",
                            "recommendation": "Provide alternative focus indicators when removing default outline",
                        }
                    )

            return {
                "compliant": len([i for i in issues if i["severity"] == "error"]) == 0,
                "issues": issues,
                "focus_styles_count": len(focus_matches),
                "outline_none_count": len(outline_none_matches),
            }

        def validate_skip_links(self, html_content: str) -> dict[str, Any]:
            """Validate presence and implementation of skip links."""
            issues = []

            # Look for skip links (usually near the top of the page)
            skip_link_patterns = [
                r'<a[^>]*href=["\']#[^"\']*["\'][^>]*>skip',
                r'<a[^>]*href=["\']#[^"\']*["\'][^>]*>jump',
            ]

            skip_links_found = []
            for pattern in skip_link_patterns:
                matches = re.findall(pattern, html_content, re.IGNORECASE)
                skip_links_found.extend(matches)

            if not skip_links_found:
                issues.append(
                    {
                        "issue": "No skip links found",
                        "severity": "error",
                        "guideline": "2.4.1",
                        "recommendation": "Add skip links to help keyboard users bypass repetitive content",
                    }
                )

            # Check if skip link targets exist
            for skip_link in skip_links_found:
                href_match = re.search(r'href=["\']#([^"\']*)["\']', skip_link)
                if href_match:
                    target_id = href_match.group(1)
                    if (
                        target_id
                        and f'id="{target_id}"' not in html_content
                        and f"id='{target_id}'" not in html_content
                    ):
                        issues.append(
                            {
                                "issue": f"Skip link target #{target_id} not found",
                                "severity": "error",
                                "guideline": "2.4.1",
                            }
                        )

            return {
                "compliant": len([i for i in issues if i["severity"] == "error"]) == 0,
                "issues": issues,
                "skip_links_found": len(skip_links_found),
            }

    return KeyboardNavigationTester()


@pytest.fixture
def screen_reader_validator():
    """Screen reader compatibility validation utilities."""

    class ScreenReaderValidator:
        def __init__(self):
            self.landmarks = [
                "banner",
                "navigation",
                "main",
                "contentinfo",
                "complementary",
                "search",
                "region",
            ]

        def validate_semantic_structure(self, html_content: str) -> dict[str, Any]:
            """Validate semantic HTML structure for screen readers."""
            issues = []

            # Check for main landmark
            main_pattern = r'<main[^>]*>|role=["\']main["\']'
            if not re.search(main_pattern, html_content, re.IGNORECASE):
                issues.append(
                    {
                        "issue": "No main landmark found",
                        "severity": "error",
                        "guideline": "1.3.1",
                        "recommendation": "Add <main> element or role='main' to identify main content",
                    }
                )

            # Check for navigation landmarks
            nav_pattern = r'<nav[^>]*>|role=["\']navigation["\']'
            nav_matches = re.findall(nav_pattern, html_content, re.IGNORECASE)

            # Check for multiple nav elements without aria-label
            if len(nav_matches) > 1:
                labeled_nav_pattern = (
                    r"<nav[^>]*aria-label[^>]*>|<nav[^>]*aria-labelledby[^>]*>"
                )
                labeled_navs = re.findall(
                    labeled_nav_pattern, html_content, re.IGNORECASE
                )

                if len(labeled_navs) < len(nav_matches):
                    issues.append(
                        {
                            "issue": "Multiple navigation landmarks without unique labels",
                            "severity": "warning",
                            "guideline": "2.4.1",
                            "recommendation": "Add aria-label or aria-labelledby to distinguish navigation areas",
                        }
                    )

            # Check for list markup for menus
            list_in_nav_pattern = r"<nav[^>]*>.*?<ul|<ol"
            if nav_matches and not re.search(
                list_in_nav_pattern, html_content, re.IGNORECASE | re.DOTALL
            ):
                issues.append(
                    {
                        "issue": "Navigation content not marked up as lists",
                        "severity": "warning",
                        "guideline": "1.3.1",
                        "recommendation": "Use <ul> or <ol> elements for navigation menus",
                    }
                )

            return {
                "compliant": len([i for i in issues if i["severity"] == "error"]) == 0,
                "issues": issues,
                "nav_count": len(nav_matches),
            }

        def validate_form_accessibility(self, html_content: str) -> dict[str, Any]:
            """Validate form accessibility for screen readers."""
            issues = []

            # Find all form inputs
            input_pattern = r'<input[^>]*type=["\'](?!hidden)[^"\']*["\'][^>]*>'
            inputs = re.findall(input_pattern, html_content, re.IGNORECASE)

            for input_elem in inputs:
                # Check for associated label
                id_match = re.search(r'id=["\']([^"\']*)["\']', input_elem)
                if id_match:
                    input_id = id_match.group(1)
                    label_pattern = f"<label[^>]*for=[\"']?{input_id}[\"']?[^>]*>"
                    if (
                        not re.search(label_pattern, html_content, re.IGNORECASE)
                        and "aria-label" not in input_elem
                        and "aria-labelledby" not in input_elem
                    ):
                        issues.append(
                            {
                                "issue": f"Input element without associated label: {input_elem[:50]}...",
                                "severity": "error",
                                "guideline": "3.3.2",
                            }
                        )
                else:
                    # Input without ID - check for aria-label
                    if "aria-label" not in input_elem:
                        issues.append(
                            {
                                "issue": f"Input element without ID or aria-label: {input_elem[:50]}...",
                                "severity": "error",
                                "guideline": "3.3.2",
                            }
                        )

            # Check for fieldset/legend for grouped inputs
            radio_pattern = r'<input[^>]*type=["\']radio["\'][^>]*>'
            checkbox_pattern = r'<input[^>]*type=["\']checkbox["\'][^>]*>'

            radio_inputs = re.findall(radio_pattern, html_content, re.IGNORECASE)
            checkbox_inputs = re.findall(checkbox_pattern, html_content, re.IGNORECASE)

            if (
                len(radio_inputs) > 1 or len(checkbox_inputs) > 2
            ) and "<fieldset" not in html_content:
                issues.append(
                    {
                        "issue": "Related form controls not grouped with fieldset/legend",
                        "severity": "warning",
                        "guideline": "1.3.1",
                        "recommendation": "Use fieldset and legend to group related form controls",
                    }
                )

            return {
                "compliant": len([i for i in issues if i["severity"] == "error"]) == 0,
                "issues": issues,
                "_total_inputs": len(inputs),
                "radio_inputs": len(radio_inputs),
                "checkbox_inputs": len(checkbox_inputs),
            }

    return ScreenReaderValidator()


@pytest.fixture
def mock_axe_core():
    """Mock axe-core accessibility testing engine."""
    axe = MagicMock()

    def mock_run_axe(
        _html_content: str, _options: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Mock axe-core analysis."""
        return {
            "violations": [
                {
                    "id": "color-contrast",
                    "impact": "serious",
                    "description": "Elements must have sufficient color contrast",
                    "help": "Ensure sufficient contrast between foreground and background colors",
                    "nodes": [
                        {
                            "html": "<div style='color: #777; background: #fff;'>Low contrast text</div>",
                            "target": ["div"],
                            "failureSummary": "Fix any of the following:\n  Element has insufficient color contrast",
                        }
                    ],
                },
                {
                    "id": "missing-alt-text",
                    "impact": "critical",
                    "description": "Images must have alternate text",
                    "help": "Images must have alternate text",
                    "nodes": [
                        {
                            "html": "<img src='test.jpg'>",
                            "target": ["img"],
                            "failureSummary": "Fix any of the following:\n  Element does not have an alt attribute",
                        }
                    ],
                },
            ],
            "passes": [
                {
                    "id": "landmark-one-main",
                    "description": "Document should have one main landmark",
                    "help": "Document should have one main landmark",
                    "nodes": [
                        {
                            "html": "<main>Content</main>",
                            "target": ["main"],
                        }
                    ],
                },
            ],
            "incomplete": [],
            "inapplicable": [],
            "url": "http://test.example.com",
            "timestamp": "2024-01-01T00:00:00.000Z",
        }

    axe.run = MagicMock(side_effect=mock_run_axe)
    return axe


@pytest.fixture
def accessibility_test_data():
    """Provide test data for accessibility testing."""
    return {
        "valid_html": """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <title>Accessible Page</title>
        </head>
        <body>
            <a href="#main-content" class="skip-link">Skip to main content</a>
            <nav aria-label="Main navigation">
                <ul>
                    <li><a href="/">Home</a></li>
                    <li><a href="/about">About</a></li>
                </ul>
            </nav>
            <main id="main-content">
                <h1>Page Title</h1>
                <p>Content goes here.</p>
                <img src="test.jpg" alt="Descriptive alt text">
                <form>
                    <label for="name">Name:</label>
                    <input type="text" id="name" required>
                    <button type="submit">Submit</button>
                </form>
            </main>
        </body>
        </html>
        """,
        "invalid_html": """
        <html>
        <head><title>Bad Page</title></head>
        <body>
            <div role="button">Clickable div</div>
            <img src="test.jpg">
            <input type="text" placeholder="Name">
            <h3>Skipped heading level</h3>
        </body>
        </html>
        """,
        "color_combinations": [
            {
                "foreground": "#000000",
                "background": "#FFFFFF",
                "expected_compliant": True,
            },
            {
                "foreground": "#777777",
                "background": "#FFFFFF",
                "expected_compliant": False,
            },
            {
                "foreground": "#FFFFFF",
                "background": "#0066CC",
                "expected_compliant": True,
            },
            {
                "foreground": "#999999",
                "background": "#CCCCCC",
                "expected_compliant": False,
            },
        ],
    }


# Pytest markers for accessibility test categorization
def pytest_configure(config):
    """Configure accessibility testing markers."""
    config.addinivalue_line("markers", "accessibility: mark test as accessibility test")
    config.addinivalue_line("markers", "a11y: mark test as general accessibility test")
    config.addinivalue_line("markers", "wcag: mark test as WCAG compliance test")
    config.addinivalue_line("markers", "screen_reader: mark test as screen reader test")
    config.addinivalue_line(
        "markers", "keyboard_navigation: mark test as keyboard navigation test"
    )
    config.addinivalue_line(
        "markers", "color_contrast: mark test as color contrast test"
    )
    config.addinivalue_line("markers", "aria: mark test as ARIA attributes test")
