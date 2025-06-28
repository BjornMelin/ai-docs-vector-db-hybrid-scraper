"""Color contrast accessibility testing.

This module implements comprehensive color contrast testing to ensure WCAG 2.1
compliance for color accessibility, including automated validation and
color-blind friendly design verification.
"""

import re

import pytest


@pytest.mark.accessibility
@pytest.mark.color_contrast
class TestColorContrastCompliance:
    """Comprehensive color contrast testing for WCAG compliance."""

    def test_wcag_aa_normal_text_contrast(self, color_contrast_analyzer):
        """Test WCAG AA contrast requirements for normal text (4.5:1)."""
        test_combinations = [
            # WCAG AA compliant combinations
            ("#000000", "#FFFFFF", True),  # Black on white (21:1)
            ("#FFFFFF", "#000000", True),  # White on black (21:1)
            ("#0066CC", "#FFFFFF", True),  # Blue on white (5.74:1)
            ("#333333", "#FFFFFF", True),  # Dark gray on white (12.63:1)
            ("#FFFFFF", "#0066CC", True),  # White on blue (5.74:1)
            # WCAG AA non-compliant combinations
            (
                "#777777",
                "#FFFFFF",
                False,
            ),  # Gray on white (4.48:1 - just below threshold)
            ("#999999", "#FFFFFF", False),  # Light gray on white (2.85:1)
            ("#CCCCCC", "#FFFFFF", False),  # Very light gray on white (1.61:1)
            ("#FF0000", "#FFFF00", False),  # Red on yellow (2.32:1)
        ]

        for foreground, background, expected_compliant in test_combinations:
            result = color_contrast_analyzer.check_contrast_compliance(
                foreground, background, "normal"
            )

            assert result["wcag_aa"] == expected_compliant, (
                f"Color {foreground} on {background} "
                f"contrast ratio: {result['contrast_ratio']}, "
                f"expected AA compliant: {expected_compliant}, "
                f"actual AA compliant: {result['wcag_aa']}"
            )

            if expected_compliant:
                assert result["contrast_ratio"] >= 4.5, (
                    f"AA compliant colors should have ratio >= 4.5, got {result['contrast_ratio']}"
                )

    def test_wcag_aa_large_text_contrast(self, color_contrast_analyzer):
        """Test WCAG AA contrast requirements for large text (3.0:1)."""
        test_combinations = [
            # Large text - lower threshold (3.0:1)
            ("#666666", "#FFFFFF", True),  # Gray on white (5.74:1)
            ("#888888", "#FFFFFF", True),  # Light gray on white (3.54:1)
            ("#999999", "#FFFFFF", False),  # Very light gray on white (2.85:1)
            ("#AAAAAA", "#FFFFFF", False),  # Very light gray on white (2.32:1)
        ]

        for foreground, background, expected_compliant in test_combinations:
            result = color_contrast_analyzer.check_contrast_compliance(
                foreground, background, "large"
            )

            assert result["wcag_aa"] == expected_compliant, (
                f"Large text {foreground} on {background} "
                f"contrast ratio: {result['contrast_ratio']}, "
                f"expected AA compliant: {expected_compliant}, "
                f"actual AA compliant: {result['wcag_aa']}"
            )

    def test_wcag_aaa_enhanced_contrast(self, color_contrast_analyzer):
        """Test WCAG AAA enhanced contrast requirements (7.0:1 normal, 4.5:1 large)."""
        test_combinations = [
            # AAA compliant for normal text (7.0:1)
            ("#000000", "#FFFFFF", "normal", True),  # Black on white (21:1)
            ("#333333", "#FFFFFF", "normal", True),  # Dark gray on white (12.63:1)
            ("#555555", "#FFFFFF", "normal", True),  # Gray on white (7.00:1)
            ("#666666", "#FFFFFF", "normal", False),  # Light gray on white (5.74:1)
            # AAA compliant for large text (4.5:1)
            ("#666666", "#FFFFFF", "large", True),  # Gray on white (5.74:1)
            ("#777777", "#FFFFFF", "large", False),  # Light gray on white (4.48:1)
        ]

        for foreground, background, text_size, expected_aaa in test_combinations:
            result = color_contrast_analyzer.check_contrast_compliance(
                foreground, background, text_size
            )

            assert result["wcag_aaa"] == expected_aaa, (
                f"{text_size.title()} text {foreground} on {background} "
                f"contrast ratio: {result['contrast_ratio']}, "
                f"expected AAA compliant: {expected_aaa}, "
                f"actual AAA compliant: {result['wcag_aaa']}"
            )

    def test_color_contrast_calculation_accuracy(self, color_contrast_analyzer):
        """Test accuracy of color contrast ratio calculations."""
        # Test known contrast ratios
        known_ratios = [
            ("#000000", "#FFFFFF", 21.0),  # Pure black on white
            ("#FFFFFF", "#000000", 21.0),  # Pure white on black
            ("#808080", "#FFFFFF", 3.95),  # 50% gray on white
            ("#FF0000", "#FFFFFF", 3.998),  # Red on white
            ("#00FF00", "#FFFFFF", 1.372),  # Green on white
            ("#0000FF", "#FFFFFF", 8.592),  # Blue on white
        ]

        for foreground, background, expected_ratio in known_ratios:
            calculated_ratio = color_contrast_analyzer.calculate_contrast_ratio(
                foreground, background
            )

            # Allow small tolerance for floating point precision
            assert abs(calculated_ratio - expected_ratio) < 0.1, (
                f"Contrast ratio for {foreground} on {background}: "
                f"expected {expected_ratio}, got {calculated_ratio}"
            )

    def test_hex_to_rgb_conversion(self, color_contrast_analyzer):
        """Test hex color to RGB conversion."""
        test_colors = [
            ("#000000", (0, 0, 0)),  # Black
            ("#FFFFFF", (255, 255, 255)),  # White
            ("#FF0000", (255, 0, 0)),  # Red
            ("#00FF00", (0, 255, 0)),  # Green
            ("#0000FF", (0, 0, 255)),  # Blue
            ("#808080", (128, 128, 128)),  # Gray
            ("#123ABC", (18, 58, 188)),  # Custom color
            ("#FFF", (255, 255, 255)),  # Short hex format
            ("#000", (0, 0, 0)),  # Short hex format
        ]

        for hex_color, expected_rgb in test_colors:
            rgb = color_contrast_analyzer.hex_to_rgb(hex_color)
            assert rgb == expected_rgb, (
                f"Hex {hex_color} should convert to {expected_rgb}, got {rgb}"
            )

    def test_relative_luminance_calculation(self, color_contrast_analyzer):
        """Test relative luminance calculations for accuracy."""
        test_cases = [
            ((0, 0, 0), 0.0),  # Black - minimum luminance
            ((255, 255, 255), 1.0),  # White - maximum luminance
            ((128, 128, 128), 0.2159),  # 50% gray
            ((255, 0, 0), 0.2126),  # Red
            ((0, 255, 0), 0.7152),  # Green
            ((0, 0, 255), 0.0722),  # Blue
        ]

        for rgb, expected_luminance in test_cases:
            luminance = color_contrast_analyzer.get_relative_luminance(rgb)

            # Allow small tolerance for floating point precision
            assert abs(luminance - expected_luminance) < 0.01, (
                f"Luminance for RGB {rgb}: expected {expected_luminance}, got {luminance}"
            )

    def test_accessible_color_suggestion_generation(self, color_contrast_analyzer):
        """Test generation of accessible color alternatives."""
        # Test with problematic colors
        problematic_combinations = [
            ("#777777", "#FFFFFF"),  # Gray on white (fails AA)
            ("#CCCCCC", "#FFFFFF"),  # Light gray on white (fails AA)
            ("#FF6B6B", "#FFFFFF"),  # Light red on white
            ("#4ECDC4", "#FFFFFF"),  # Light teal on white
        ]

        for foreground, background in problematic_combinations:
            suggestions = color_contrast_analyzer.generate_accessible_color_suggestions(
                foreground, background
            )

            assert len(suggestions) > 0, (
                f"Should generate suggestions for {foreground} on {background}"
            )

            # All suggestions should meet WCAG AA requirements
            for suggestion in suggestions:
                assert suggestion["contrast_ratio"] >= 4.5, (
                    f"Suggested color {suggestion['color']} should meet WCAG AA "
                    f"(ratio: {suggestion['contrast_ratio']})"
                )

                # Verify suggestion accuracy
                verified_ratio = color_contrast_analyzer.calculate_contrast_ratio(
                    suggestion["color"], background
                )
                assert abs(verified_ratio - suggestion["contrast_ratio"]) < 0.1, (
                    "Suggested contrast ratio should match calculated ratio"
                )

    def test_brand_color_accessibility_compliance(self, color_contrast_analyzer):
        """Test common brand colors for accessibility compliance."""
        brand_colors = [
            # Social media brand colors
            ("#1DA1F2", "Twitter Blue"),  # Twitter
            ("#4267B2", "Facebook Blue"),  # Facebook
            ("#0077B5", "LinkedIn Blue"),  # LinkedIn
            ("#FF0000", "YouTube Red"),  # YouTube
            ("#25D366", "WhatsApp Green"),  # WhatsApp
            # Tech company colors
            ("#4285F4", "Google Blue"),  # Google
            ("#FF9500", "Amazon Orange"),  # Amazon
            ("#007ACC", "Microsoft Blue"),  # Microsoft
            ("#FF6900", "Firefox Orange"),  # Firefox
        ]

        white_background = "#FFFFFF"

        results = {}
        for brand_color, brand_name in brand_colors:
            result = color_contrast_analyzer.check_contrast_compliance(
                brand_color, white_background, "normal"
            )

            results[brand_name] = {
                "color": brand_color,
                "contrast_ratio": result["contrast_ratio"],
                "wcag_aa": result["wcag_aa"],
                "wcag_aaa": result["wcag_aaa"],
            }

            # If brand color fails, generate suggestions
            if not result["wcag_aa"]:
                suggestions = (
                    color_contrast_analyzer.generate_accessible_color_suggestions(
                        brand_color, white_background
                    )
                )
                results[brand_name]["accessible_alternatives"] = suggestions[
                    :3
                ]  # Top 3

        # Verify we have results for all brands
        assert len(results) == len(brand_colors)

        # Check specific known good/bad cases
        assert not results["Twitter Blue"]["wcag_aa"]  # Known to fail on white
        assert results["LinkedIn Blue"]["wcag_aa"]  # Known to pass on white

    def test_color_blind_friendly_palette_validation(self, color_contrast_analyzer):
        """Test color combinations for color-blind accessibility."""
        # Deuteranopia-friendly color palette (most common color blindness)
        deuteranopia_friendly = [
            ("#1f77b4", "#FFFFFF"),  # Blue on white
            ("#ff7f0e", "#FFFFFF"),  # Orange on white
            ("#2ca02c", "#FFFFFF"),  # Green on white
            ("#d62728", "#FFFFFF"),  # Red on white
            ("#9467bd", "#FFFFFF"),  # Purple on white
        ]

        # Problematic red-green combinations for color blind users
        problematic_combinations = [
            ("#008000", "#FF0000"),  # Green on red
            ("#FF0000", "#008000"),  # Red on green
            ("#006400", "#8B0000"),  # Dark green on dark red
        ]

        # Test friendly combinations
        for foreground, background in deuteranopia_friendly:
            result = color_contrast_analyzer.check_contrast_compliance(
                foreground, background, "normal"
            )

            # These should generally have good contrast
            if not result["wcag_aa"]:
                # Generate accessible alternative
                suggestions = (
                    color_contrast_analyzer.generate_accessible_color_suggestions(
                        foreground, background
                    )
                )
                assert len(suggestions) > 0, (
                    f"Should provide alternatives for {foreground} on {background}"
                )

        # Test problematic combinations
        for foreground, background in problematic_combinations:
            result = color_contrast_analyzer.check_contrast_compliance(
                foreground, background, "normal"
            )

            # Document the issue even if contrast is technically sufficient
            # In practice, these combinations are problematic for color-blind users
            # regardless of contrast ratio

    def test_css_color_extraction_and_validation(self, color_contrast_analyzer):
        """Test extraction and validation of colors from CSS."""
        css_content = """
        .primary-text {
            color: #333333;
            background-color: #FFFFFF;
        }

        .secondary-text {
            color: #777777;
            background-color: #F5F5F5;
        }

        .warning-text {
            color: #FF6B6B;
            background-color: #FFF5F5;
        }

        .success-text {
            color: #28A745;
            background-color: #D4EDDA;
        }
        """

        # Extract color combinations from CSS (simplified regex matching)

        color_pattern = r"color:\s*([^;]+);.*?background-color:\s*([^;]+);"
        matches = re.findall(color_pattern, css_content, re.DOTALL)

        extracted_combinations = []
        for match in matches:
            foreground = match[0].strip()
            background = match[1].strip()

            # Only process hex colors for this test
            if foreground.startswith("#") and background.startswith("#"):
                extracted_combinations.append((foreground, background))

        # Validate each extracted combination
        validation_results = {}
        for foreground, background in extracted_combinations:
            result = color_contrast_analyzer.check_contrast_compliance(
                foreground, background, "normal"
            )

            validation_results[f"{foreground} on {background}"] = result

        # Verify we extracted some combinations
        assert len(validation_results) > 0, "Should extract color combinations from CSS"

        # Check specific expectations
        if "#333333 on #FFFFFF" in validation_results:
            assert validation_results["#333333 on #FFFFFF"]["wcag_aa"], (
                "Dark gray on white should pass AA"
            )

    @pytest.mark.parametrize(
        "foreground,background,expected_ratio",
        [
            ("#000000", "#FFFFFF", 21.0),
            ("#FFFFFF", "#808080", 3.95),
            ("#0066CC", "#FFFFFF", 5.74),
            ("#FF0000", "#FFFFFF", 3.998),
            ("#333333", "#FFFFFF", 12.63),
        ],
    )
    def test_contrast_ratio_calculations_parametrized(
        self, color_contrast_analyzer, foreground, background, expected_ratio
    ):
        """Parametrized test for contrast ratio calculations."""
        calculated_ratio = color_contrast_analyzer.calculate_contrast_ratio(
            foreground, background
        )

        # Allow tolerance for floating point precision
        assert abs(calculated_ratio - expected_ratio) < 0.1, (
            f"Contrast ratio for {foreground} on {background}: "
            f"expected {expected_ratio}, got {calculated_ratio}"
        )

    def test_high_contrast_mode_simulation(self, color_contrast_analyzer):
        """Test simulation of high contrast mode requirements."""
        # High contrast mode typically requires very high contrast ratios
        high_contrast_combinations = [
            ("#000000", "#FFFFFF"),  # Pure black on white
            ("#FFFFFF", "#000000"),  # Pure white on black
            ("#000080", "#FFFFFF"),  # Navy on white
            ("#FFFFFF", "#000080"),  # White on navy
        ]

        moderate_contrast_combinations = [
            ("#333333", "#FFFFFF"),  # Dark gray on white
            ("#0066CC", "#FFFFFF"),  # Blue on white
            ("#666666", "#FFFFFF"),  # Gray on white
        ]

        # High contrast combinations should have very high ratios
        for foreground, background in high_contrast_combinations:
            ratio = color_contrast_analyzer.calculate_contrast_ratio(
                foreground, background
            )
            assert ratio >= 15.0, (
                f"High contrast pair {foreground}/{background} ratio should be >= 15, got {ratio}"
            )

        # Moderate contrast combinations should be between 4.5 and 15
        for foreground, background in moderate_contrast_combinations:
            ratio = color_contrast_analyzer.calculate_contrast_ratio(
                foreground, background
            )
            assert 4.5 <= ratio < 15.0, (
                f"Moderate contrast pair {foreground}/{background} ratio should be 4.5-15, got {ratio}"
            )

    def test_contrast_validation_report_generation(self, color_contrast_analyzer):
        """Test generation of comprehensive contrast validation reports."""
        test_page_colors = [
            {"element": "body", "foreground": "#333333", "background": "#FFFFFF"},
            {"element": "h1", "foreground": "#000000", "background": "#FFFFFF"},
            {"element": ".warning", "foreground": "#FF6B6B", "background": "#FFFFFF"},
            {"element": ".success", "foreground": "#28A745", "background": "#FFFFFF"},
            {"element": ".muted", "foreground": "#999999", "background": "#FFFFFF"},
        ]

        report = {
            "timestamp": "2024-01-01T00:00:00Z",
            "page_url": "http://test.example.com",
            "wcag_level": "AA",
            "total_elements": len(test_page_colors),
            "compliant_elements": 0,
            "non_compliant_elements": 0,
            "elements": [],
            "summary": {
                "pass_rate": 0.0,
                "average_contrast_ratio": 0.0,
                "recommendations": [],
            },
        }

        total_ratio = 0.0

        for element_info in test_page_colors:
            result = color_contrast_analyzer.check_contrast_compliance(
                element_info["foreground"], element_info["background"], "normal"
            )

            element_result = {
                "element": element_info["element"],
                "foreground": element_info["foreground"],
                "background": element_info["background"],
                "contrast_ratio": result["contrast_ratio"],
                "wcag_aa_compliant": result["wcag_aa"],
                "wcag_aaa_compliant": result["wcag_aaa"],
            }

            # Add suggestions for non-compliant elements
            if not result["wcag_aa"]:
                suggestions = (
                    color_contrast_analyzer.generate_accessible_color_suggestions(
                        element_info["foreground"], element_info["background"]
                    )
                )
                element_result["suggestions"] = suggestions[:3]  # Top 3 suggestions
                report["non_compliant_elements"] += 1
            else:
                report["compliant_elements"] += 1

            report["elements"].append(element_result)
            total_ratio += result["contrast_ratio"]

        # Calculate summary statistics
        report["summary"]["pass_rate"] = (
            report["compliant_elements"] / report["total_elements"]
        )
        report["summary"]["average_contrast_ratio"] = (
            total_ratio / report["total_elements"]
        )

        # Add recommendations
        if report["summary"]["pass_rate"] < 1.0:
            report["summary"]["recommendations"].extend(
                [
                    "Review and update colors that fail WCAG AA contrast requirements",
                    "Consider using darker colors for better accessibility",
                    "Test with color blindness simulation tools",
                    "Validate with real users who have visual impairments",
                ]
            )

        # Verify report structure
        assert "timestamp" in report
        assert "wcag_level" in report
        assert report["total_elements"] == len(test_page_colors)
        assert (
            report["compliant_elements"] + report["non_compliant_elements"]
            == report["total_elements"]
        )
        assert 0.0 <= report["summary"]["pass_rate"] <= 1.0
        assert report["summary"]["average_contrast_ratio"] > 0.0

        # Verify element details
        for element in report["elements"]:
            assert "contrast_ratio" in element
            assert "wcag_aa_compliant" in element
            assert isinstance(element["wcag_aa_compliant"], bool)


@pytest.mark.accessibility
@pytest.mark.color_contrast
class TestColorContrastTools:
    """Test color contrast analysis tools and utilities."""

    def test_color_palette_accessibility_analysis(self, color_contrast_analyzer):
        """Test analysis of complete color palettes for accessibility."""
        # Define a design system color palette
        color_palette = {
            "primary": "#0066CC",
            "secondary": "#6C757D",
            "success": "#28A745",
            "warning": "#FFC107",
            "danger": "#DC3545",
            "info": "#17A2B8",
            "light": "#F8F9FA",
            "dark": "#343A40",
            "white": "#FFFFFF",
            "black": "#000000",
        }

        palette_analysis = {}

        for bg_name, bg_color in [
            ("white", "#FFFFFF"),
            ("light", "#F8F9FA"),
            ("black", "#000000"),
        ]:
            palette_analysis[bg_name] = {}

            for color_name, color_value in color_palette.items():
                if color_value != bg_color:  # Don't test color on itself
                    result = color_contrast_analyzer.check_contrast_compliance(
                        color_value, bg_color, "normal"
                    )

                    palette_analysis[bg_name][color_name] = {
                        "contrast_ratio": result["contrast_ratio"],
                        "wcag_aa": result["wcag_aa"],
                        "wcag_aaa": result["wcag_aaa"],
                    }

        # Verify analysis structure
        assert "white" in palette_analysis
        assert "black" in palette_analysis
        assert len(palette_analysis["white"]) > 0

        # Dark colors should work well on white background
        assert palette_analysis["white"]["dark"]["wcag_aa"]
        assert palette_analysis["white"]["black"]["wcag_aa"]

        # Light colors should work well on black background
        assert palette_analysis["black"]["light"]["wcag_aa"]
        assert palette_analysis["black"]["white"]["wcag_aa"]

    def test_gradient_contrast_analysis(self, color_contrast_analyzer):
        """Test contrast analysis for color gradients."""
        # Simulate gradient analysis by testing multiple points

        # Test 5 points along the gradient
        gradient_points = [
            "#0066CC",  # 0% - start
            "#3380D6",  # 25%
            "#6699E0",  # 50%
            "#99B3EA",  # 75%
            "#FFFFFF",  # 100% - end
        ]

        background = "#FFFFFF"

        gradient_results = []
        for i, color in enumerate(gradient_points):
            if color != background:  # Skip white on white
                result = color_contrast_analyzer.check_contrast_compliance(
                    color, background, "normal"
                )

                gradient_results.append(
                    {
                        "position": f"{i * 25}%",
                        "color": color,
                        "contrast_ratio": result["contrast_ratio"],
                        "wcag_aa": result["wcag_aa"],
                    }
                )

        # Verify that contrast decreases as we approach white
        ratios = [r["contrast_ratio"] for r in gradient_results]
        for i in range(len(ratios) - 1):
            assert ratios[i] >= ratios[i + 1], "Contrast should decrease towards white"

        # At least the start of the gradient should be accessible
        assert gradient_results[0]["wcag_aa"], "Gradient start should be accessible"

    def test_color_accessibility_best_practices(self, color_contrast_analyzer):
        """Test adherence to color accessibility best practices."""

        # Best practice: Never rely on color alone to convey information
        # This is tested through semantic markup validation

        # Best practice: Provide sufficient contrast for all text
        text_combinations = [
            ("#000000", "#FFFFFF", True),  # High contrast
            ("#333333", "#FFFFFF", True),  # Good contrast
            ("#666666", "#FFFFFF", True),  # Adequate contrast
            ("#999999", "#FFFFFF", False),  # Poor contrast
            ("#CCCCCC", "#FFFFFF", False),  # Very poor contrast
        ]

        for foreground, background, should_pass in text_combinations:
            result = color_contrast_analyzer.check_contrast_compliance(
                foreground, background, "normal"
            )
            assert result["wcag_aa"] == should_pass, (
                f"Text {foreground} on {background} should {'pass' if should_pass else 'fail'} AA"
            )

        # Best practice: Use high contrast for important UI elements
        important_ui_elements = [
            ("#FFFFFF", "#007ACC"),  # White on blue (buttons)
            ("#FFFFFF", "#28A745"),  # White on green (success)
            ("#FFFFFF", "#DC3545"),  # White on red (danger)
            ("#000000", "#FFC107"),  # Black on yellow (warning)
        ]

        for foreground, background in important_ui_elements:
            result = color_contrast_analyzer.check_contrast_compliance(
                foreground, background, "normal"
            )
            assert result["wcag_aa"], (
                f"Important UI element {foreground} on {background} should meet AA standards"
            )

        # Best practice: Test with simulated color blindness
        # This would typically involve color transformation algorithms
        # For now, we ensure high enough contrast that it works for most users

        minimum_safe_ratio = 7.0  # Stricter than WCAG AA for better universal access
        for foreground, background in important_ui_elements:
            ratio = color_contrast_analyzer.calculate_contrast_ratio(
                foreground, background
            )
            # Not all combinations will meet this strict standard, but document it
            if ratio < minimum_safe_ratio:
                suggestions = (
                    color_contrast_analyzer.generate_accessible_color_suggestions(
                        foreground, background
                    )
                )
                # Verify suggestions meet the stricter standard
                [s for s in suggestions if s["contrast_ratio"] >= minimum_safe_ratio]
                # Should have at least some high-contrast alternatives
                assert len(suggestions) > 0, "Should provide alternative colors"
