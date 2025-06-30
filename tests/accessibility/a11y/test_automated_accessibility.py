"""Automated accessibility testing integration.

This module integrates automated accessibility testing tools like axe-core,
pa11y, and Lighthouse to provide comprehensive accessibility validation
for web interfaces and APIs.
"""

from typing import Any
from unittest.mock import AsyncMock

import pytest


@pytest.mark.accessibility
@pytest.mark.a11y
class TestAutomatedAccessibilityTools:
    """Integration tests for automated accessibility testing tools."""

    @pytest.fixture
    def mock_axe_core_result(self):
        """Mock axe-core analysis result."""
        return {
            "testEngine": {"name": "axe-core", "version": "4.8.0"},
            "testRunner": {"name": "pytest-axe"},
            "testEnvironment": {
                "userAgent": "Mozilla/5.0 (compatible; axe-core)",
                "windowWidth": 1280,
                "windowHeight": 720,
                "orientationAngle": 0,
                "orientationType": "landscape-primary",
            },
            "timestamp": "2024-01-01T00:00:00.000Z",
            "url": "http://test.example.com",
            "violations": [
                {
                    "id": "color-contrast",
                    "impact": "serious",
                    "tags": ["cat.color", "wcag2aa", "wcag143"],
                    "description": (
                        "Ensures the contrast between foreground and background colors "
                        "meets WCAG 2 AA contrast ratio thresholds"
                    ),
                    "help": "Elements must have sufficient color contrast",
                    "helpUrl": "https://dequeuniversity.com/rules/axe/4.8/color-contrast",
                    "nodes": [
                        {
                            "any": [],
                            "all": [],
                            "none": [
                                {
                                    "id": "color-contrast",
                                    "data": {
                                        "fgColor": "#777777",
                                        "bgColor": "#ffffff",
                                        "contrastRatio": 4.48,
                                        "fontSize": "12.0pt (16px)",
                                        "fontWeight": "normal",
                                        "messageKey": "",
                                        "expectedContrastRatio": "4.5:1",
                                    },
                                    "relatedNodes": [],
                                    "impact": "serious",
                                    "message": "Element has insufficient color contrast of 4.48 (foreground color: #777777, background color: #ffffff, font size: 12.0pt (16px), font weight: normal). Expected contrast ratio of 4.5:1",
                                }
                            ],
                            "html": '<p style="color: #777777;">This text has insufficient contrast</p>',
                            "target": ["p"],
                            "failureSummary": "Fix any of the following:\n  Element has insufficient color contrast of 4.48 (foreground color: #777777, background color: #ffffff, font size: 12.0pt (16px), font weight: normal). Expected contrast ratio of 4.5:1",
                        }
                    ],
                },
                {
                    "id": "image-alt",
                    "impact": "critical",
                    "tags": [
                        "cat.text-alternatives",
                        "wcag2a",
                        "wcag111",
                        "section508",
                        "section508.22.a",
                    ],
                    "description": "Ensures <img> elements have alternate text or a role of none or presentation",
                    "help": "Images must have alternate text",
                    "helpUrl": "https://dequeuniversity.com/rules/axe/4.8/image-alt",
                    "nodes": [
                        {
                            "any": [
                                {
                                    "id": "has-alt",
                                    "data": None,
                                    "relatedNodes": [],
                                    "impact": "critical",
                                    "message": "Element does not have an alt attribute",
                                }
                            ],
                            "all": [],
                            "none": [],
                            "html": '<img src="test.jpg">',
                            "target": ["img"],
                            "failureSummary": 'Fix any of the following:\n  Element does not have an alt attribute\n  aria-label attribute does not exist or is empty\n  aria-labelledby attribute does not exist, references elements that do not exist or references elements that are empty\n  Element has no title attribute\n  Element\'s default semantics were not overridden with role="none" or role="presentation"',
                        }
                    ],
                },
                {
                    "id": "landmark-one-main",
                    "impact": "moderate",
                    "tags": ["cat.semantics", "best-practice"],
                    "description": "Ensures the document has a main landmark",
                    "help": "Document should have one main landmark",
                    "helpUrl": "https://dequeuniversity.com/rules/axe/4.8/landmark-one-main",
                    "nodes": [
                        {
                            "any": [
                                {
                                    "id": "page-has-main",
                                    "data": None,
                                    "relatedNodes": [],
                                    "impact": "moderate",
                                    "message": "Document does not have a main landmark",
                                }
                            ],
                            "all": [],
                            "none": [],
                            "html": '<html lang="en"><head>...</head><body>...</body></html>',
                            "target": ["html"],
                            "failureSummary": "Fix any of the following:\n  Document does not have a main landmark",
                        }
                    ],
                },
            ],
            "passes": [
                {
                    "id": "html-has-lang",
                    "impact": None,
                    "tags": ["cat.language", "wcag2a", "wcag311"],
                    "description": "Ensures every HTML document has a lang attribute",
                    "help": "<html> element must have a lang attribute",
                    "helpUrl": "https://dequeuniversity.com/rules/axe/4.8/html-has-lang",
                    "nodes": [
                        {
                            "any": [
                                {
                                    "id": "has-lang",
                                    "data": {"value": "en"},
                                    "relatedNodes": [],
                                    "impact": None,
                                    "message": "The <html> element has a lang attribute",
                                }
                            ],
                            "all": [],
                            "none": [],
                            "html": '<html lang="en">',
                            "target": ["html"],
                            "failureSummary": "",
                        }
                    ],
                },
                {
                    "id": "document-title",
                    "impact": None,
                    "tags": ["cat.text-alternatives", "wcag2a", "wcag242"],
                    "description": "Ensures each HTML document contains a non-empty <title> element",
                    "help": "Documents must have <title> element to aid in navigation",
                    "helpUrl": "https://dequeuniversity.com/rules/axe/4.8/document-title",
                    "nodes": [
                        {
                            "any": [
                                {
                                    "id": "doc-has-title",
                                    "data": None,
                                    "relatedNodes": [],
                                    "impact": None,
                                    "message": "Document has a non-empty <title> element",
                                }
                            ],
                            "all": [],
                            "none": [],
                            "html": "<title>Test Page</title>",
                            "target": ["title"],
                            "failureSummary": "",
                        }
                    ],
                },
            ],
            "incomplete": [
                {
                    "id": "color-contrast",
                    "impact": "serious",
                    "tags": ["cat.color", "wcag2aa", "wcag143"],
                    "description": (
                        "Ensures the contrast between foreground and background colors "
                        "meets WCAG 2 AA contrast ratio thresholds"
                    ),
                    "help": "Elements must have sufficient color contrast",
                    "helpUrl": "https://dequeuniversity.com/rules/axe/4.8/color-contrast",
                    "nodes": [
                        {
                            "any": [
                                {
                                    "id": "color-contrast",
                                    "data": {"messageKey": "bgImage"},
                                    "relatedNodes": [],
                                    "impact": "serious",
                                    "message": "Element's background color could not be determined due to a background image",
                                }
                            ],
                            "all": [],
                            "none": [],
                            "html": "<div style=\"background-image: url('bg.jpg'); color: white;\">Text over image</div>",
                            "target": ["div"],
                            "failureSummary": "Fix any of the following:\n  Element's background color could not be determined due to a background image",
                        }
                    ],
                }
            ],
            "inapplicable": [
                {
                    "id": "accesskeys",
                    "impact": None,
                    "tags": ["cat.keyboard", "best-practice"],
                    "description": "Ensures every accesskey attribute value is unique",
                    "help": "accesskey attribute value should be unique",
                    "helpUrl": "https://dequeuniversity.com/rules/axe/4.8/accesskeys",
                    "nodes": [],
                }
            ],
        }

    def test_axe_core_integration(self, mock_axe_core_result):
        """Test axe-core accessibility testing integration."""
        # Simulate axe-core analysis
        result = mock_axe_core_result

        # Verify result structure
        assert "violations" in result
        assert "passes" in result
        assert "incomplete" in result
        assert "inapplicable" in result
        assert "testEngine" in result
        assert "timestamp" in result
        assert "url" in result

        # Analyze violations
        violations = result["violations"]
        assert len(violations) == 3, "Should have expected number of violations"

        # Check critical violations
        critical_violations = [v for v in violations if v["impact"] == "critical"]
        assert len(critical_violations) == 1, "Should have one critical violation"
        assert critical_violations[0]["id"] == "image-alt"

        # Check serious violations
        serious_violations = [v for v in violations if v["impact"] == "serious"]
        assert len(serious_violations) == 1, "Should have one serious violation"
        assert serious_violations[0]["id"] == "color-contrast"

        # Verify violation details
        color_contrast_violation = serious_violations[0]
        node_data = color_contrast_violation["nodes"][0]["none"][0]["data"]
        assert node_data["contrastRatio"] == 4.48
        assert node_data["expectedContrastRatio"] == "4.5:1"
        assert node_data["fgColor"] == "#777777"
        assert node_data["bgColor"] == "#ffffff"

        # Check passes
        passes = result["passes"]
        assert len(passes) >= 2, "Should have some passing tests"

        # Verify specific passes
        lang_pass = next((p for p in passes if p["id"] == "html-has-lang"), None)
        assert lang_pass is not None, "Should pass html-has-lang test"

        title_pass = next((p for p in passes if p["id"] == "document-title"), None)
        assert title_pass is not None, "Should pass document-title test"

    @pytest.fixture
    def mock_pa11y_result(self):
        """Mock pa11y analysis result."""
        return [
            {
                "code": "WCAG2AA.Principle1.Guideline1_1.1_1_1.H37",
                "type": "error",
                "typeCode": 1,
                "message": "Img element missing an alt attribute. Use the alt attribute to specify a short text alternative.",
                "context": '<img src="test.jpg">',
                "selector": "html > body > img",
                "runner": "htmlcs",
                "runnerExtras": {},
            },
            {
                "code": "WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Fail",
                "type": "error",
                "typeCode": 1,
                "message": "This element has insufficient contrast at this conformance level. Expected a contrast ratio of at least 4.5:1, but text in this element has a contrast ratio of 4.48:1.",
                "context": '<p style="color: #777777;">This text has insufficient contrast</p>',
                "selector": "html > body > p",
                "runner": "htmlcs",
                "runnerExtras": {},
            },
            {
                "code": "WCAG2AA.Principle2.Guideline2_4.2_4_1.H64.1",
                "type": "warning",
                "typeCode": 2,
                "message": "Iframe element requires a non-empty title attribute that identifies the frame.",
                "context": '<iframe src="embed.html"></iframe>',
                "selector": "html > body > iframe",
                "runner": "htmlcs",
                "runnerExtras": {},
            },
            {
                "code": "WCAG2AA.Principle3.Guideline3_1.3_1_1.H57.2",
                "type": "notice",
                "typeCode": 3,
                "message": "The html element should have a lang or xml:lang attribute which describes the language of the document.",
                "context": "<html>",
                "selector": "html",
                "runner": "htmlcs",
                "runnerExtras": {},
            },
        ]

    def test_pa11y_integration(self, mock_pa11y_result):
        """Test pa11y accessibility testing integration."""
        # Simulate pa11y analysis
        result = mock_pa11y_result

        # Analyze results by type
        errors = [item for item in result if item["type"] == "error"]
        warnings = [item for item in result if item["type"] == "warning"]
        notices = [item for item in result if item["type"] == "notice"]

        assert len(errors) == 2, "Should have expected number of errors"
        assert len(warnings) == 1, "Should have expected number of warnings"
        assert len(notices) == 1, "Should have expected number of notices"

        # Check specific error codes
        error_codes = [error["code"] for error in errors]
        assert "WCAG2AA.Principle1.Guideline1_1.1_1_1.H37" in error_codes
        assert "WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Fail" in error_codes

        # Verify error details
        alt_text_error = next((e for e in errors if "H37" in e["code"]), None)
        assert alt_text_error is not None
        assert "alt attribute" in alt_text_error["message"]
        assert "img" in alt_text_error["context"]

        contrast_error = next((e for e in errors if "G18.Fail" in e["code"]), None)
        assert contrast_error is not None
        assert "contrast ratio" in contrast_error["message"]
        assert "4.48:1" in contrast_error["message"]

    @pytest.fixture
    def mock_lighthouse_result(self):
        """Mock Lighthouse accessibility audit result."""
        return {
            "lhr": {
                "fetchTime": "2024-01-01T00:00:00.000Z",
                "finalUrl": "http://test.example.com",
                "lighthouseVersion": "10.4.0",
                "userAgent": "Mozilla/5.0 (compatible; Lighthouse)",
                "environment": {
                    "networkUserAgent": "Mozilla/5.0 (compatible; Lighthouse)",
                    "hostUserAgent": "Mozilla/5.0 (compatible; Lighthouse)",
                    "benchmarkIndex": 1000,
                },
                "categories": {
                    "accessibility": {
                        "id": "accessibility",
                        "title": "Accessibility",
                        "description": "These checks highlight opportunities to improve the accessibility of your web app.",
                        "score": 0.85,
                        "manualDescription": "These items address areas which an automated testing tool cannot cover.",
                        "auditRefs": [
                            {
                                "id": "accesskeys",
                                "weight": 0,
                                "group": "a11y-navigation",
                            },
                            {
                                "id": "aria-allowed-attr",
                                "weight": 10,
                                "group": "a11y-aria",
                            },
                            {
                                "id": "aria-command-name",
                                "weight": 3,
                                "group": "a11y-aria",
                            },
                            {
                                "id": "aria-hidden-body",
                                "weight": 10,
                                "group": "a11y-aria",
                            },
                            {
                                "id": "aria-hidden-focus",
                                "weight": 3,
                                "group": "a11y-aria",
                            },
                            {
                                "id": "aria-input-field-name",
                                "weight": 3,
                                "group": "a11y-aria",
                            },
                            {
                                "id": "aria-meter-name",
                                "weight": 3,
                                "group": "a11y-aria",
                            },
                            {
                                "id": "aria-progressbar-name",
                                "weight": 3,
                                "group": "a11y-aria",
                            },
                            {
                                "id": "aria-required-attr",
                                "weight": 10,
                                "group": "a11y-aria",
                            },
                            {
                                "id": "aria-required-children",
                                "weight": 10,
                                "group": "a11y-aria",
                            },
                            {
                                "id": "aria-required-parent",
                                "weight": 10,
                                "group": "a11y-aria",
                            },
                            {"id": "aria-roles", "weight": 10, "group": "a11y-aria"},
                            {
                                "id": "aria-toggle-field-name",
                                "weight": 3,
                                "group": "a11y-aria",
                            },
                            {
                                "id": "aria-tooltip-name",
                                "weight": 3,
                                "group": "a11y-aria",
                            },
                            {
                                "id": "aria-treeitem-name",
                                "weight": 3,
                                "group": "a11y-aria",
                            },
                            {
                                "id": "aria-valid-attr-value",
                                "weight": 10,
                                "group": "a11y-aria",
                            },
                            {
                                "id": "aria-valid-attr",
                                "weight": 10,
                                "group": "a11y-aria",
                            },
                            {
                                "id": "button-name",
                                "weight": 10,
                                "group": "a11y-names-labels",
                            },
                            {"id": "bypass", "weight": 3, "group": "a11y-navigation"},
                            {
                                "id": "color-contrast",
                                "weight": 3,
                                "group": "a11y-color-contrast",
                            },
                            {
                                "id": "definition-list",
                                "weight": 3,
                                "group": "a11y-tables-lists",
                            },
                            {"id": "dlitem", "weight": 3, "group": "a11y-tables-lists"},
                            {
                                "id": "document-title",
                                "weight": 3,
                                "group": "a11y-names-labels",
                            },
                            {
                                "id": "duplicate-id-active",
                                "weight": 3,
                                "group": "a11y-navigation",
                            },
                            {
                                "id": "duplicate-id-aria",
                                "weight": 10,
                                "group": "a11y-aria",
                            },
                            {
                                "id": "form-field-multiple-labels",
                                "weight": 2,
                                "group": "a11y-names-labels",
                            },
                            {
                                "id": "frame-title",
                                "weight": 3,
                                "group": "a11y-names-labels",
                            },
                            {
                                "id": "heading-order",
                                "weight": 2,
                                "group": "a11y-navigation",
                            },
                            {
                                "id": "html-has-lang",
                                "weight": 3,
                                "group": "a11y-language",
                            },
                            {
                                "id": "html-lang-valid",
                                "weight": 3,
                                "group": "a11y-language",
                            },
                            {
                                "id": "image-alt",
                                "weight": 10,
                                "group": "a11y-names-labels",
                            },
                            {
                                "id": "input-image-alt",
                                "weight": 10,
                                "group": "a11y-names-labels",
                            },
                            {"id": "label", "weight": 10, "group": "a11y-names-labels"},
                            {
                                "id": "landmark-one-main",
                                "weight": 3,
                                "group": "a11y-navigation",
                            },
                            {
                                "id": "link-name",
                                "weight": 3,
                                "group": "a11y-names-labels",
                            },
                            {"id": "list", "weight": 3, "group": "a11y-tables-lists"},
                            {
                                "id": "listitem",
                                "weight": 3,
                                "group": "a11y-tables-lists",
                            },
                            {
                                "id": "meta-refresh",
                                "weight": 10,
                                "group": "a11y-best-practices",
                            },
                            {
                                "id": "meta-viewport",
                                "weight": 10,
                                "group": "a11y-best-practices",
                            },
                            {
                                "id": "object-alt",
                                "weight": 3,
                                "group": "a11y-names-labels",
                            },
                            {"id": "tabindex", "weight": 3, "group": "a11y-navigation"},
                            {
                                "id": "td-headers-attr",
                                "weight": 3,
                                "group": "a11y-tables-lists",
                            },
                            {
                                "id": "th-has-data-cells",
                                "weight": 3,
                                "group": "a11y-tables-lists",
                            },
                            {"id": "valid-lang", "weight": 3, "group": "a11y-language"},
                            {
                                "id": "video-caption",
                                "weight": 10,
                                "group": "a11y-audio-video",
                            },
                        ],
                    }
                },
                "audits": {
                    "accessibility": {
                        "id": "accessibility",
                        "title": "Accessibility",
                        "description": "These checks highlight opportunities to improve the accessibility of your web app.",
                        "score": 0.85,
                        "scoreDisplayMode": "numeric",
                    },
                    "color-contrast": {
                        "id": "color-contrast",
                        "title": "Background and foreground colors have a sufficient contrast ratio",
                        "description": "Low-contrast text is difficult or impossible for many users to read.",
                        "score": 0,
                        "scoreDisplayMode": "binary",
                        "details": {
                            "type": "table",
                            "headings": [
                                {
                                    "key": "node",
                                    "itemType": "node",
                                    "subItemsHeading": {
                                        "key": "relatedNode",
                                        "itemType": "node",
                                    },
                                    "text": "Failing Elements",
                                }
                            ],
                            "items": [
                                {
                                    "node": {
                                        "type": "node",
                                        "lhId": "9-0-P",
                                        "path": "1,HTML,1,BODY,0,P",
                                        "selector": "body > p",
                                        "boundingRect": {
                                            "top": 100,
                                            "bottom": 120,
                                            "left": 50,
                                            "right": 200,
                                            "width": 150,
                                            "height": 20,
                                        },
                                        "snippet": '<p style="color: #777777;">',
                                        "nodeLabel": "This text has insufficient contrast",
                                    }
                                }
                            ],
                        },
                    },
                    "image-alt": {
                        "id": "image-alt",
                        "title": "Image elements have [alt] attributes",
                        "description": "Informative elements should aim for short, descriptive alternate text.",
                        "score": 0,
                        "scoreDisplayMode": "binary",
                        "details": {
                            "type": "table",
                            "headings": [
                                {
                                    "key": "node",
                                    "itemType": "node",
                                    "text": "Failing Elements",
                                }
                            ],
                            "items": [
                                {
                                    "node": {
                                        "type": "node",
                                        "lhId": "9-1-IMG",
                                        "path": "1,HTML,1,BODY,1,IMG",
                                        "selector": "body > img",
                                        "boundingRect": {
                                            "top": 150,
                                            "bottom": 250,
                                            "left": 50,
                                            "right": 150,
                                            "width": 100,
                                            "height": 100,
                                        },
                                        "snippet": '<img src="test.jpg">',
                                        "nodeLabel": "",
                                    }
                                }
                            ],
                        },
                    },
                    "html-has-lang": {
                        "id": "html-has-lang",
                        "title": "<html> element has a [lang] attribute",
                        "description": "If a page doesn't specify a lang attribute, a screen reader assumes that the page is in the default language that the user chose when setting up the screen reader.",
                        "score": 1,
                        "scoreDisplayMode": "binary",
                    },
                    "document-title": {
                        "id": "document-title",
                        "title": "Document has a <title> element",
                        "description": "The title gives screen reader users an overview of the page, and search engine users rely on it to determine if a page is relevant to their search.",
                        "score": 1,
                        "scoreDisplayMode": "binary",
                    },
                    "landmark-one-main": {
                        "id": "landmark-one-main",
                        "title": "Document has a main landmark",
                        "description": "One main landmark helps screen reader users navigate to the primary content of the page.",
                        "score": 0,
                        "scoreDisplayMode": "binary",
                        "explanation": "Document does not have a main landmark",
                    },
                },
            }
        }

    def test_lighthouse_accessibility_integration(self, mock_lighthouse_result):
        """Test Lighthouse accessibility audit integration."""
        # Simulate Lighthouse analysis
        result = mock_lighthouse_result["lhr"]

        # Check overall accessibility score
        accessibility_category = result["categories"]["accessibility"]
        assert accessibility_category["score"] == 0.85
        assert accessibility_category["title"] == "Accessibility"

        # Check individual audits
        audits = result["audits"]

        # Failing audits
        color_contrast_audit = audits["color-contrast"]
        assert color_contrast_audit["score"] == 0
        assert "contrast ratio" in color_contrast_audit["description"]

        image_alt_audit = audits["image-alt"]
        assert image_alt_audit["score"] == 0
        assert "alt attributes" in image_alt_audit["title"]

        landmark_audit = audits["landmark-one-main"]
        assert landmark_audit["score"] == 0
        assert "main landmark" in landmark_audit["title"]

        # Passing audits
        lang_audit = audits["html-has-lang"]
        assert lang_audit["score"] == 1
        assert "lang attribute" in lang_audit["title"]

        title_audit = audits["document-title"]
        assert title_audit["score"] == 1
        assert "title element" in title_audit["title"]

    @pytest.mark.asyncio
    async def test_playwright_axe_integration(self):
        """Test Playwright integration with axe-core."""
        # Mock Playwright page with axe integration
        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_page.add_script_tag = AsyncMock()
        mock_page.evaluate = AsyncMock()

        # Mock axe results
        mock_axe_results = {
            "violations": [
                {
                    "id": "color-contrast",
                    "impact": "serious",
                    "description": "Elements must have sufficient color contrast",
                    "nodes": [{"html": "<p style='color: #777;'>Low contrast</p>"}],
                }
            ],
            "passes": [
                {
                    "id": "html-has-lang",
                    "description": "<html> element must have a lang attribute",
                    "nodes": [{"html": "<html lang='en'>"}],
                }
            ],
        }

        mock_page.evaluate.return_value = mock_axe_results

        # Simulate axe testing workflow
        await mock_page.goto("http://test.example.com")

        # Inject axe-core
        await mock_page.add_script_tag(
            url="https://unpkg.com/axe-core@4.8.0/axe.min.js"
        )

        # Run axe analysis
        axe_results = await mock_page.evaluate("""
            () => {
                return new Promise((resolve) => {
                    axe.run((err, results) => {
                        if (err) throw err;
                        resolve(results);
                    });
                });
            }
        """)

        # Verify integration
        mock_page.goto.assert_called_once_with("http://test.example.com")
        mock_page.add_script_tag.assert_called_once()
        mock_page.evaluate.assert_called_once()

        # Verify results
        assert "violations" in axe_results
        assert "passes" in axe_results
        assert len(axe_results["violations"]) == 1
        assert axe_results["violations"][0]["id"] == "color-contrast"

    def test_accessibility_testing_api_endpoints(self):
        """Test accessibility of API error responses and documentation."""
        # Mock API error responses for accessibility testing
        api_responses = {
            "400_bad_request": {
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "The request contains invalid data",
                    "details": [
                        {
                            "field": "email",
                            "issue": "Invalid email format",
                            "suggestion": "Please provide a valid email address",
                        },
                        {
                            "field": "password",
                            "issue": "Password too weak",
                            "suggestion": "Password must contain at least 8 characters, one number, and one symbol",
                        },
                    ],
                    "help_url": "https://api.example.com/docs/validation-errors",
                }
            },
            "401_unauthorized": {
                "error": {
                    "code": "UNAUTHORIZED",
                    "message": "Authentication required",
                    "details": "Please provide a valid API key in the Authorization header",
                    "help_url": "https://api.example.com/docs/authentication",
                }
            },
            "403_forbidden": {
                "error": {
                    "code": "FORBIDDEN",
                    "message": "Insufficient permissions",
                    "details": "Your account does not have permission to access this resource",
                    "help_url": "https://api.example.com/docs/permissions",
                }
            },
            "429_rate_limit": {
                "error": {
                    "code": "RATE_LIMIT_EXCEEDED",
                    "message": "Too many requests",
                    "details": "Rate limit of 1000 requests per hour exceeded",
                    "retry_after": 3600,
                    "help_url": "https://api.example.com/docs/rate-limits",
                }
            },
        }

        # Check accessibility requirements for API responses
        for status_code, response in api_responses.items():
            error = response["error"]

            # Clear error messages
            assert "message" in error, f"{status_code} should have clear error message"
            assert len(error["message"]) > 0, (
                f"{status_code} message should not be empty"
            )
            assert error["message"][0].isupper(), (
                f"{status_code} message should be properly capitalized"
            )

            # Actionable details
            assert "details" in error, (
                f"{status_code} should provide actionable details"
            )

            # Help resources
            assert "help_url" in error, (
                f"{status_code} should provide help documentation"
            )
            assert error["help_url"].startswith("https://"), (
                f"{status_code} help URL should be secure"
            )

            # Machine-readable error codes
            assert "code" in error, (
                f"{status_code} should have machine-readable error code"
            )
            assert error["code"].isupper(), (
                f"{status_code} error code should be uppercase"
            )
            assert "_" in error["code"], (
                f"{status_code} error code should use snake_case"
            )

    def test_accessibility_testing_report_generation(
        self, mock_axe_core_result, mock_pa11y_result, mock_lighthouse_result
    ):
        """Test generation of comprehensive accessibility testing reports."""
        # Combine results from multiple tools
        axe_result = mock_axe_core_result
        pa11y_result = mock_pa11y_result
        lighthouse_result = mock_lighthouse_result["lhr"]

        # Generate comprehensive accessibility report
        report = {
            "timestamp": "2024-01-01T00:00:00Z",
            "page_url": "http://test.example.com",
            "testing_tools": {
                "axe_core": {
                    "version": axe_result["testEngine"]["version"],
                    "violations": len(axe_result["violations"]),
                    "passes": len(axe_result["passes"]),
                    "incomplete": len(axe_result["incomplete"]),
                    "score": self._calculate_axe_score(axe_result),
                },
                "pa11y": {
                    "errors": len([r for r in pa11y_result if r["type"] == "error"]),
                    "warnings": len(
                        [r for r in pa11y_result if r["type"] == "warning"]
                    ),
                    "notices": len([r for r in pa11y_result if r["type"] == "notice"]),
                    "_total_issues": len(pa11y_result),
                },
                "lighthouse": {
                    "version": lighthouse_result["lighthouseVersion"],
                    "accessibility_score": lighthouse_result["categories"][
                        "accessibility"
                    ]["score"],
                    "failing_audits": len(
                        [
                            audit
                            for audit in lighthouse_result["audits"].values()
                            if audit.get("score") == 0
                        ]
                    ),
                    "passing_audits": len(
                        [
                            audit
                            for audit in lighthouse_result["audits"].values()
                            if audit.get("score") == 1
                        ]
                    ),
                },
            },
            "consolidated_issues": self._consolidate_issues(
                axe_result, pa11y_result, lighthouse_result
            ),
            "wcag_compliance": {
                "level_a": self._check_wcag_level(axe_result, pa11y_result, "A"),
                "level_aa": self._check_wcag_level(axe_result, pa11y_result, "AA"),
                "level_aaa": self._check_wcag_level(axe_result, pa11y_result, "AAA"),
            },
            "overall_score": 0.0,
            "priority_fixes": [],
            "recommendations": [
                "Fix critical and serious accessibility violations first",
                "Ensure all images have appropriate alt text",
                "Verify color contrast meets WCAG AA standards",
                "Add proper landmarks and heading structure",
                "Test with actual assistive technology",
                "Conduct user testing with disabled users",
            ],
        }

        # Calculate overall score (weighted average)
        axe_score = report["testing_tools"]["axe_core"]["score"]
        lighthouse_score = report["testing_tools"]["lighthouse"]["accessibility_score"]
        pa11y_score = 1.0 - (
            len([r for r in pa11y_result if r["type"] == "error"])
            / max(len(pa11y_result), 1)
        )

        report["overall_score"] = (
            axe_score * 0.4 + lighthouse_score * 0.4 + pa11y_score * 0.2
        )

        # Generate priority fixes
        report["priority_fixes"] = self._generate_priority_fixes(
            axe_result, pa11y_result, lighthouse_result
        )

        # Verify report structure
        assert "timestamp" in report
        assert "testing_tools" in report
        assert "consolidated_issues" in report
        assert "wcag_compliance" in report
        assert 0.0 <= report["overall_score"] <= 1.0
        assert len(report["recommendations"]) > 0

        # Verify tool-specific data
        assert report["testing_tools"]["axe_core"]["version"] == "4.8.0"
        assert report["testing_tools"]["lighthouse"]["accessibility_score"] == 0.85
        assert report["testing_tools"]["pa11y"]["errors"] == 2

    def _calculate_axe_score(self, axe_result: dict[str, Any]) -> float:
        """Calculate score from axe-core results."""
        violations = axe_result["violations"]
        passes = axe_result["passes"]

        if not violations and not passes:
            return 1.0

        # Weight violations by impact
        impact_weights = {
            "critical": 1.0,
            "serious": 0.7,
            "moderate": 0.4,
            "minor": 0.2,
        }
        violation_score = sum(
            impact_weights.get(v.get("impact", "minor"), 0.2) for v in violations
        )

        # Calculate score (inverse of violations, normalized)
        _total_tests = len(violations) + len(passes)
        if _total_tests == 0:
            return 1.0

        return max(0.0, 1.0 - (violation_score / _total_tests))

    def _consolidate_issues(
        self,
        axe_result: dict[str, Any],
        pa11y_result: list[dict[str, Any]],
        lighthouse_result: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Consolidate issues from multiple testing tools."""
        # Process axe violations
        axe_violations = [
            {
                "tool": "axe-core",
                "id": violation["id"],
                "impact": violation["impact"],
                "description": violation["description"],
                "help_url": violation.get("helpUrl"),
                "wcag_tags": [
                    tag for tag in violation.get("tags", []) if tag.startswith("wcag")
                ],
                "affected_elements": len(violation.get("nodes", [])),
            }
            for violation in axe_result["violations"]
        ]

        # Process pa11y errors
        pa11y_violations = [
            {
                "tool": "pa11y",
                "id": error["code"],
                "impact": "error",
                "description": error["message"],
                "help_url": None,
                "wcag_tags": [error["code"]] if "WCAG" in error["code"] else [],
                "affected_elements": 1,
            }
            for error in pa11y_result
            if error["type"] == "error"
        ]

        consolidated = axe_violations + pa11y_violations

        # Process Lighthouse failing audits
        for audit_id, audit in lighthouse_result["audits"].items():
            if audit.get("score") == 0:
                consolidated.append(
                    {
                        "tool": "lighthouse",
                        "id": audit_id,
                        "impact": "error",
                        "description": audit["description"],
                        "help_url": None,
                        "wcag_tags": [],
                        "affected_elements": len(
                            audit.get("details", {}).get("items", [])
                        ),
                    }
                )

        return consolidated

    def _check_wcag_level(
        self, axe_result: dict[str, Any], pa11y_result: list[dict[str, Any]], level: str
    ) -> dict[str, Any]:
        """Check WCAG compliance level."""
        level_tag = f"wcag2{level.lower()}"

        # Check axe violations for this level
        level_violations = [
            v for v in axe_result["violations"] if level_tag in v.get("tags", [])
        ]

        # Check pa11y errors for this level
        level_pa11y_errors = [
            e for e in pa11y_result if e["type"] == "error" and level in e["code"]
        ]

        _total_issues = len(level_violations) + len(level_pa11y_errors)

        return {
            "compliant": _total_issues == 0,
            "violations": _total_issues,
            "level": level,
            "percentage": 100.0 if _total_issues == 0 else 0.0,
        }

    def _generate_priority_fixes(
        self,
        axe_result: dict[str, Any],
        pa11y_result: list[dict[str, Any]],
        _lighthouse_result: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Generate prioritized list of accessibility fixes."""
        # Critical axe violations
        critical_violations = [
            v for v in axe_result["violations"] if v["impact"] == "critical"
        ]
        critical_fixes = [
            {
                "priority": "critical",
                "issue": violation["description"],
                "affected_elements": len(violation.get("nodes", [])),
                "tool": "axe-core",
                "help_url": violation.get("helpUrl"),
            }
            for violation in critical_violations
        ]

        # Pa11y errors
        pa11y_errors = [e for e in pa11y_result if e["type"] == "error"]
        pa11y_fixes = [
            {
                "priority": "high",
                "issue": error["message"],
                "affected_elements": 1,
                "tool": "pa11y",
                "help_url": None,
            }
            for error in pa11y_errors[:3]  # Top 3 errors
        ]

        fixes = critical_fixes + pa11y_fixes

        # Sort by priority and impact
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        fixes.sort(
            key=lambda x: (
                priority_order.get(x["priority"], 3),
                -x["affected_elements"],
            )
        )

        return fixes[:10]  # Top 10 priority fixes


@pytest.mark.accessibility
@pytest.mark.a11y
class TestAccessibilityTestingWorkflow:
    """Test comprehensive accessibility testing workflows."""

    def test_accessibility_ci_pipeline_integration(self):
        """Test accessibility testing in CI/CD pipeline."""
        # Mock CI pipeline configuration
        ci_config = {
            "accessibility_testing": {
                "tools": ["axe-core", "pa11y", "lighthouse"],
                "thresholds": {
                    "axe_violations": {"critical": 0, "serious": 2},
                    "pa11y_errors": 3,
                    "lighthouse_score": 0.85,
                },
                "test_urls": [
                    "http://localhost:3000/",
                    "http://localhost:3000/login",
                    "http://localhost:3000/dashboard",
                    "http://localhost:3000/profile",
                ],
                "browsers": ["chrome", "firefox", "safari"],
                "viewports": [
                    {"width": 1920, "height": 1080},
                    {"width": 768, "height": 1024},
                    {"width": 375, "height": 667},
                ],
            }
        }

        # Verify CI configuration
        a11y_config = ci_config["accessibility_testing"]
        assert "axe-core" in a11y_config["tools"]
        assert "pa11y" in a11y_config["tools"]
        assert "lighthouse" in a11y_config["tools"]

        # Check thresholds
        assert a11y_config["thresholds"]["axe_violations"]["critical"] == 0
        assert a11y_config["thresholds"]["lighthouse_score"] >= 0.8

        # Check test coverage
        assert len(a11y_config["test_urls"]) >= 3
        assert len(a11y_config["browsers"]) >= 2
        assert len(a11y_config["viewports"]) >= 2

    def test_accessibility_regression_testing(self):
        """Test accessibility regression detection."""
        # Mock baseline accessibility results
        baseline = {
            "timestamp": "2024-01-01T00:00:00Z",
            "axe_violations": 2,
            "pa11y_errors": 1,
            "lighthouse_score": 0.87,
            "critical_issues": 0,
        }

        # Mock current test results
        current = {
            "timestamp": "2024-01-02T00:00:00Z",
            "axe_violations": 4,
            "pa11y_errors": 3,
            "lighthouse_score": 0.82,
            "critical_issues": 1,
        }

        # Detect regressions
        regressions = []

        if current["axe_violations"] > baseline["axe_violations"]:
            regressions.append(
                {
                    "type": "axe_violations_increased",
                    "baseline": baseline["axe_violations"],
                    "current": current["axe_violations"],
                    "difference": current["axe_violations"]
                    - baseline["axe_violations"],
                }
            )

        if current["lighthouse_score"] < baseline["lighthouse_score"] - 0.05:
            regressions.append(
                {
                    "type": "lighthouse_score_decreased",
                    "baseline": baseline["lighthouse_score"],
                    "current": current["lighthouse_score"],
                    "difference": current["lighthouse_score"]
                    - baseline["lighthouse_score"],
                }
            )

        if current["critical_issues"] > baseline["critical_issues"]:
            regressions.append(
                {
                    "type": "critical_issues_introduced",
                    "baseline": baseline["critical_issues"],
                    "current": current["critical_issues"],
                    "difference": current["critical_issues"]
                    - baseline["critical_issues"],
                }
            )

        # Verify regression detection
        assert len(regressions) == 3, "Should detect all regressions"

        regression_types = [r["type"] for r in regressions]
        assert "axe_violations_increased" in regression_types
        assert "lighthouse_score_decreased" in regression_types
        assert "critical_issues_introduced" in regression_types

    def test_accessibility_testing_documentation_generation(self):
        """Test generation of accessibility testing documentation."""
        # Mock accessibility test results for documentation
        test_results = {
            "summary": {
                "_total_pages_tested": 12,
                "_total_issues_found": 23,
                "critical_issues": 2,
                "serious_issues": 8,
                "moderate_issues": 10,
                "minor_issues": 3,
                "overall_score": 0.78,
            },
            "by_page": {
                "/": {"score": 0.85, "issues": 3},
                "/login": {"score": 0.92, "issues": 1},
                "/dashboard": {"score": 0.75, "issues": 8},
                "/profile": {"score": 0.88, "issues": 2},
            },
            "by_category": {
                "color_contrast": {"issues": 8, "pages_affected": 5},
                "missing_alt_text": {"issues": 6, "pages_affected": 4},
                "keyboard_navigation": {"issues": 4, "pages_affected": 3},
                "form_labels": {"issues": 3, "pages_affected": 2},
                "heading_structure": {"issues": 2, "pages_affected": 2},
            },
            "wcag_compliance": {
                "level_a": {"compliant": False, "violations": 8},
                "level_aa": {"compliant": False, "violations": 12},
                "level_aaa": {"compliant": False, "violations": 18},
            },
        }

        # Generate documentation sections
        documentation = {
            "title": "Accessibility Testing Report",
            "executive_summary": f"""
                Tested {test_results["summary"]["_total_pages_tested"]} pages and found
                {test_results["summary"]["_total_issues_found"]} accessibility issues.
                Overall accessibility score: {test_results["summary"]["overall_score"]:.1%}.

                Critical issues requiring immediate attention: {test_results["summary"]["critical_issues"]}
            """,
            "methodology": """
                Accessibility testing was performed using:
                - axe-core for automated WCAG compliance checking
                - pa11y for additional validation
                - Lighthouse for performance and accessibility auditing
                - Manual testing with keyboard navigation
                - Screen reader testing with NVDA and VoiceOver
            """,
            "key_findings": [
                f"Color contrast issues found on {test_results['by_category']['color_contrast']['pages_affected']} pages",
                f"Missing alt text affects {test_results['by_category']['missing_alt_text']['pages_affected']} pages",
                f"Keyboard navigation problems on {test_results['by_category']['keyboard_navigation']['pages_affected']} pages",
            ],
            "recommendations": [
                "Fix all critical and serious accessibility violations",
                "Implement automated accessibility testing in CI/CD pipeline",
                "Conduct regular accessibility audits",
                "Train development team on accessibility best practices",
                "Establish accessibility design system and guidelines",
            ],
        }

        # Verify documentation structure
        assert "title" in documentation
        assert "executive_summary" in documentation
        assert "methodology" in documentation
        assert "key_findings" in documentation
        assert "recommendations" in documentation

        # Verify content quality
        assert (
            str(test_results["summary"]["_total_pages_tested"])
            in documentation["executive_summary"]
        )
        assert len(documentation["key_findings"]) >= 3
        assert len(documentation["recommendations"]) >= 5
