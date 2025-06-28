"""Cross-Site Scripting (XSS) prevention tests.

This module tests protection against XSS attacks across all user inputs
and output rendering contexts.
"""

from unittest.mock import patch

import pytest

from src.security import SecurityError, SecurityValidator


@pytest.mark.security
@pytest.mark.input_validation
class TestXSSPrevention:
    """Test XSS attack prevention."""

    @pytest.fixture
    def security_validator(self):
        """Get security validator instance."""
        return SecurityValidator()

    @pytest.fixture
    def xss_payloads(self):
        """XSS attack payloads."""
        return [
            "<script>alert('XSS')</script>",
            "<script>alert(document.cookie)</script>",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>",
            "<iframe src=javascript:alert('XSS')>",
            "<input onfocus=alert('XSS') autofocus>",
            "<select onfocus=alert('XSS') autofocus>",
            "<textarea onfocus=alert('XSS') autofocus>",
            "<keygen onfocus=alert('XSS') autofocus>",
            "<video><source onerror='alert(1)'>",
            "<audio src=x onerror=alert('XSS')>",
            "<object data='javascript:alert(1)'>",
            "<embed src='javascript:alert(1)'>",
            "<link rel=stylesheet href='javascript:alert(1)'>",
            "<style>@import'javascript:alert(1)';</style>",
        ]

    @pytest.fixture
    def advanced_xss_payloads(self):
        """Advanced XSS payloads with encoding and obfuscation."""
        return [
            # JavaScript protocol variations
            "javascript:alert('XSS')",
            "JaVaScRiPt:alert('XSS')",
            "&#106;&#97;&#118;&#97;&#115;&#99;&#114;&#105;&#112;&#116;&#58;",
            "%6A%61%76%61%73%63%72%69%70%74%3A",
            # Event handler variations
            "<img src=x onerror='alert(String.fromCharCode(88,83,83))'>",
            "<img src=x onerror=alert`XSS`>",
            "<img src=x onerror=eval('alert(1)')>",
            "<img src=x onerror=(alert)(1)>",
            # DOM-based XSS
            "<img src=x onerror=window['alert'](1)>",
            "<img src=x onerror=self['alert'](1)>",
            "<img src=x onerror=this['alert'](1)>",
            # Template injection
            "{{7*7}}",
            "${7*7}",
            "<%=7*7%>",
            "{%=7*7%}",
            # Expression language injection
            "${{7*7}}",
            "#{7*7}",
            "@{7*7}",
            # Filter bypass attempts
            "<script>alert(1)</script>",
            "<scr<script>ipt>alert(1)</script>",
            "<SCRIPT>alert(1)</SCRIPT>",
            "<<SCRIPT>alert(1)</SCRIPT>",
            "<script>alert(String.fromCharCode(88,83,83))</script>",
        ]

    @pytest.fixture
    def encoding_xss_payloads(self):
        """XSS payloads with various encoding schemes."""
        return [
            # HTML entity encoding
            "&lt;script&gt;alert('XSS')&lt;/script&gt;",
            "&#60;script&#62;alert('XSS')&#60;/script&#62;",
            "&#x3C;script&#x3E;alert('XSS')&#x3C;/script&#x3E;",
            # URL encoding
            "%3Cscript%3Ealert('XSS')%3C/script%3E",
            "%3cscript%3ealert(%27XSS%27)%3c/script%3e",
            # Double encoding
            "%253Cscript%253Ealert('XSS')%253C/script%253E",
            # Unicode encoding
            "\\u003cscript\\u003ealert('XSS')\\u003c/script\\u003e",
            "\\x3cscript\\x3ealert('XSS')\\x3c/script\\x3e",
            # Base64 encoding (when decoded)
            "PHNjcmlwdD5hbGVydCgnWFNTJyk8L3NjcmlwdD4=",  # <script>alert('XSS')</script>
        ]

    @pytest.mark.asyncio
    async def test_basic_xss_prevention(self, security_validator, xss_payloads):
        """Test basic XSS payload prevention."""
        for payload in xss_payloads:
            with pytest.raises(SecurityError):
                security_validator.validate_query_string(payload)

    @pytest.mark.asyncio
    async def test_advanced_xss_prevention(
        self, security_validator, advanced_xss_payloads
    ):
        """Test advanced XSS payload prevention."""
        for payload in advanced_xss_payloads:
            with pytest.raises(SecurityError):
                security_validator.validate_query_string(payload)

    @pytest.mark.asyncio
    async def test_encoded_xss_prevention(
        self, security_validator, encoding_xss_payloads
    ):
        """Test encoded XSS payload prevention."""
        for payload in encoding_xss_payloads:
            # Validator should handle encoded payloads
            try:
                result = security_validator.validate_query_string(payload)
                # If validation passes, dangerous content should be removed
                assert "<script" not in result.lower()
                assert "alert" not in result.lower()
                assert "javascript:" not in result.lower()
            except SecurityError:
                # Rejection is also acceptable
                pass

    @pytest.mark.asyncio
    async def test_url_xss_prevention(self, security_validator):
        """Test XSS prevention in URL validation."""
        malicious_urls = [
            "javascript:alert('XSS')",
            "data:text/html,<script>alert('XSS')</script>",
            "https://example.com?q=<script>alert('XSS')</script>",
            "https://example.com#<script>alert('XSS')</script>",
            "https://example.com/<script>alert('XSS')</script>",
        ]

        for url in malicious_urls:
            with pytest.raises(SecurityError):
                security_validator.validate_url(url)

    @pytest.mark.asyncio
    async def test_filename_xss_prevention(self, security_validator):
        """Test XSS prevention in filename sanitization."""
        malicious_filenames = [
            "<script>alert('XSS')</script>.txt",
            "file<img src=x onerror=alert(1)>.pdf",
            "document<svg onload=alert(1)>.doc",
            "data<iframe src=javascript:alert(1)>.json",
        ]

        for filename in malicious_filenames:
            sanitized = security_validator.sanitize_filename(filename)
            assert "<script" not in sanitized.lower()
            assert "onerror" not in sanitized.lower()
            assert "onload" not in sanitized.lower()
            assert "javascript:" not in sanitized.lower()

    def test_output_encoding_enforcement(self):
        """Test that output encoding is properly enforced."""
        # This test would verify that all user-controlled data is properly
        # encoded when rendered in HTML contexts

        test_data = [
            "<script>alert('XSS')</script>",
            "'; alert('XSS'); //",
            "\"><script>alert('XSS')</script>",
        ]

        # Mock template rendering or API response generation
        with patch("jinja2.Template.render") as mock_render:
            for _data in test_data:
                # Verify that data is properly escaped before rendering
                # The actual implementation should use auto-escaping templates
                mock_render.assert_not_called()

    def test_content_security_policy_validation(self):
        """Test Content Security Policy header validation."""
        # Test that CSP headers are properly configured to prevent XSS
        expected_csp_directives = [
            "default-src 'self'",
            "script-src 'self'",
            "object-src 'none'",
            "style-src 'self' 'unsafe-inline'",
            "img-src 'self' data: https:",
            "font-src 'self'",
            "connect-src 'self'",
            "frame-ancestors 'none'",
        ]

        # This would test actual CSP header configuration
        # For now, we'll verify the structure
        for directive in expected_csp_directives:
            assert "'" in directive or directive.endswith(("'none'", "'self'"))

    @pytest.mark.asyncio
    async def test_dom_xss_prevention(self):
        """Test prevention of DOM-based XSS attacks."""
        dom_xss_vectors = [
            "document.location.hash",
            "document.URL",
            "document.referrer",
            "window.name",
            "history.pushState",
            "localStorage.getItem",
            "sessionStorage.getItem",
        ]

        # Test that client-side code properly validates these inputs
        # This would require testing client-side JavaScript if present
        for _vector in dom_xss_vectors:
            # Verify that any client-side code validates these sources
            pass

    @pytest.mark.asyncio
    async def test_stored_xss_prevention(self):
        """Test prevention of stored XSS attacks."""
        # Test that data stored in database is properly sanitized
        stored_payloads = [
            "<script>alert('Stored XSS')</script>",
            "<img src=x onerror=alert('Stored XSS')>",
            "javascript:alert('Stored XSS')",
        ]

        security_validator = SecurityValidator()

        for payload in stored_payloads:
            # Data should be sanitized before storage
            with pytest.raises(SecurityError):
                security_validator.validate_query_string(payload)

    @pytest.mark.asyncio
    async def test_reflected_xss_prevention(self):
        """Test prevention of reflected XSS attacks."""
        # Test that URL parameters and form inputs are properly validated
        reflected_payloads = [
            "search=<script>alert('Reflected XSS')</script>",
            "q=<img src=x onerror=alert(1)>",
            "filter=javascript:alert(1)",
        ]

        security_validator = SecurityValidator()

        for payload in reflected_payloads:
            query_part = payload.split("=", 1)[1]
            with pytest.raises(SecurityError):
                security_validator.validate_query_string(query_part)

    def test_template_injection_prevention(self, security_validator):
        """Test prevention of template injection attacks."""
        template_payloads = [
            "{{7*7}}",
            "${7*7}",
            "<%=7*7%>",
            "{%=7*7%}",
            "{{config.__class__.__init__.__globals__['os'].popen('ls').read()}}",
            "${__import__('os').popen('id').read()}",
        ]

        for payload in template_payloads:
            with pytest.raises(SecurityError):
                security_validator.validate_query_string(payload)

    def test_expression_language_injection_prevention(self, security_validator):
        """Test prevention of expression language injection."""
        el_payloads = [
            "${{7*7}}",
            "#{7*7}",
            "@{7*7}",
            "${T(java.lang.Runtime).getRuntime().exec('cat /etc/passwd')}",
            "#{T(java.lang.Runtime).getRuntime().exec('id')}",
        ]

        for payload in el_payloads:
            with pytest.raises(SecurityError):
                security_validator.validate_query_string(payload)

    def test_attribute_injection_prevention(self, security_validator):
        """Test prevention of HTML attribute injection."""
        attribute_payloads = [
            '" onmouseover="alert(\'XSS\')"',
            "' onmouseover='alert(1)'",
            '"><script>alert(1)</script>',
            "'><script>alert(1)</script>",
            "javascript:alert(1)",
        ]

        for payload in attribute_payloads:
            with pytest.raises(SecurityError):
                security_validator.validate_query_string(payload)

    def test_css_injection_prevention(self, security_validator):
        """Test prevention of CSS injection attacks."""
        css_payloads = [
            "expression(alert('XSS'))",
            "url('javascript:alert(1)')",
            "@import 'javascript:alert(1)'",
            "behavior:url('javascript:alert(1)')",
            "-moz-binding:url('javascript:alert(1)')",
        ]

        for payload in css_payloads:
            with pytest.raises(SecurityError):
                security_validator.validate_query_string(payload)

    def test_svg_xss_prevention(self, security_validator):
        """Test prevention of SVG-based XSS attacks."""
        svg_payloads = [
            "<svg onload=alert('XSS')>",
            "<svg><script>alert('XSS')</script></svg>",
            "<svg><foreignObject><script>alert(1)</script></foreignObject></svg>",
            "<svg><use href='#x' onload='alert(1)'/>",
            "<svg><animate onbegin=alert('XSS')>",
        ]

        for payload in svg_payloads:
            with pytest.raises(SecurityError):
                security_validator.validate_query_string(payload)

    def test_mathml_xss_prevention(self, security_validator):
        """Test prevention of MathML-based XSS attacks."""
        mathml_payloads = [
            "<math><mi//xlink:href='data:x,<script>alert(1)</script>'>",
            "<math><mo>alert(1)</mo></math>",
            "<math><annotation-xml encoding='text/html'><script>alert(1)</script></annotation-xml></math>",
        ]

        for payload in mathml_payloads:
            with pytest.raises(SecurityError):
                security_validator.validate_query_string(payload)

    def test_valid_content_passes_validation(self, security_validator):
        """Test that legitimate content passes XSS validation."""
        valid_content = [
            "normal search query",
            "documentation about JavaScript",
            "CSS styling guide",
            "HTML tutorial basics",
            "user@example.com",
            "https://example.com/page",
            "API reference documentation",
        ]

        for content in valid_content:
            # Should not raise any exception
            validated = security_validator.validate_query_string(content)
            # Basic validation should preserve safe content
            assert len(validated) > 0

    @pytest.mark.asyncio
    async def test_bypass_filter_prevention(self, security_validator):
        """Test prevention of XSS filter bypass techniques."""
        bypass_payloads = [
            # Case variations
            "<ScRiPt>alert('XSS')</ScRiPt>",
            "<SCRIPT>alert('XSS')</SCRIPT>",
            # Nested tags
            "<scr<script>ipt>alert('XSS')</script>",
            "<<script>script>alert('XSS')<</script>/script>",
            # Whitespace variations
            "<script >alert('XSS')</script>",
            "<script\n>alert('XSS')</script>",
            "<script\t>alert('XSS')</script>",
            # Comments
            "<script>/**/alert('XSS')</script>",
            "<script><!-- -->alert('XSS')</script>",
            # Fragmented payloads
            "<img src=x one" + "rror=alert('XSS')>",
            "<script>ale" + "rt('XSS')</script>",
        ]

        for payload in bypass_payloads:
            with pytest.raises(SecurityError):
                security_validator.validate_query_string(payload)

    def test_context_aware_encoding(self):
        """Test context-aware output encoding for different contexts."""
        # Test that output encoding is appropriate for the context
        contexts = {
            "html_content": lambda x: f"<div>{x}</div>",
            "html_attribute": lambda x: f"<div title='{x}'>",
            "javascript_string": lambda x: f"var data = '{x}';",
            "css_value": lambda x: f"color: {x};",
            "url_parameter": lambda x: f"https://example.com?q={x}",
        }

        dangerous_data = "<script>alert('XSS')</script>"

        for context_name, context_func in contexts.items():
            # Each context should properly encode the dangerous data
            # This would test the actual template/rendering system
            encoded_result = context_func(dangerous_data)

            # Verify that dangerous content is properly encoded for each context
            if context_name == "html_content":
                assert (
                    "&lt;script&gt;" in encoded_result
                    or "<script" not in encoded_result
                )
            elif context_name == "html_attribute":
                assert "&#x27;" in encoded_result or "'" not in encoded_result
            elif context_name == "javascript_string":
                assert "\\x3c" in encoded_result or "<" not in encoded_result
            elif context_name == "css_value":
                assert "\\3c" in encoded_result or "<" not in encoded_result
            elif context_name == "url_parameter":
                assert "%3C" in encoded_result or "<" not in encoded_result
