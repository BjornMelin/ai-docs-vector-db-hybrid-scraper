# Accessibility Testing Guide

This guide provides comprehensive instructions for accessibility testing in the AI Documentation Vector DB Hybrid Scraper project.

## Overview

Our accessibility testing framework ensures WCAG 2.1 AA compliance across all user-facing components, including web interfaces, API responses, and documentation.

## Testing Tools

### Primary Tools
- **axe-core**: Automated accessibility testing engine
- **pa11y**: Command-line accessibility testing
- **Lighthouse**: Google's accessibility audit tool
- **Playwright**: Browser automation for accessibility testing

### Manual Testing Tools
- **NVDA**: Screen reader testing (Windows)
- **VoiceOver**: Screen reader testing (macOS)
- **JAWS**: Professional screen reader testing
- **Keyboard Navigation**: Tab order and focus testing

## Test Categories

### 1. WCAG Compliance Testing (`wcag/`)
Tests for Web Content Accessibility Guidelines 2.1 compliance:

- **Level A**: Basic accessibility requirements
- **Level AA**: Standard compliance (our target)
- **Level AAA**: Enhanced accessibility

Key test areas:
- Non-text content (alt text)
- Info and relationships (semantic HTML)
- Contrast requirements
- Keyboard accessibility
- Focus management
- Language identification
- Form labels and instructions

### 2. Color Contrast Testing (`color_contrast/`)
Ensures sufficient color contrast ratios:

- **Normal text**: 4.5:1 minimum (WCAG AA)
- **Large text**: 3.0:1 minimum (WCAG AA)
- **Enhanced contrast**: 7.0:1 (WCAG AAA)

Features:
- Automated contrast ratio calculation
- Brand color accessibility validation
- Color-blind friendly palette testing
- Accessible color suggestion generation

### 3. Keyboard Navigation Testing (`keyboard_navigation/`)
Validates keyboard accessibility:

- Tab order validation
- Focus indicator visibility
- Skip link functionality
- Keyboard trap prevention
- Interactive element accessibility

### 4. Screen Reader Testing (`screen_reader/`)
Ensures compatibility with assistive technology:

- Semantic HTML structure
- ARIA implementation
- Form accessibility
- Table accessibility
- Live region functionality
- Landmark navigation

### 5. ARIA Attributes Testing (`aria/`)
Validates ARIA implementation:

- Role validation
- Property and state management
- Labeling hierarchy
- Live regions
- Interactive widgets
- Relationship attributes

### 6. Automated Testing Integration (`a11y/`)
Integrates multiple testing tools:

- axe-core automation
- pa11y integration
- Lighthouse audit integration
- Browser automation
- CI/CD pipeline integration

## Running Tests

### Install Dependencies
```bash
# Install accessibility testing tools
uv add --group accessibility axe-core-python pytest-axe selenium playwright beautifulsoup4 lxml
```

### Basic Test Execution
```bash
# Run all accessibility tests
uv run pytest tests/accessibility/ -v

# Run specific test categories
uv run pytest tests/accessibility/wcag/ -v
uv run pytest tests/accessibility/color_contrast/ -v
uv run pytest tests/accessibility/keyboard_navigation/ -v
uv run pytest tests/accessibility/screen_reader/ -v
uv run pytest tests/accessibility/aria/ -v
uv run pytest tests/accessibility/a11y/ -v

# Run with accessibility markers
uv run pytest -m accessibility -v
uv run pytest -m wcag -v
uv run pytest -m color_contrast -v
uv run pytest -m keyboard_navigation -v
uv run pytest -m screen_reader -v
uv run pytest -m aria -v
```

### Test Configuration
Configure accessibility testing in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
    "accessibility: mark test as accessibility test",
    "a11y: mark test as general accessibility test", 
    "wcag: mark test as WCAG compliance test",
    "screen_reader: mark test as screen reader test",
    "keyboard_navigation: mark test as keyboard navigation test",
    "color_contrast: mark test as color contrast test",
    "aria: mark test as ARIA attributes test",
]
```

## Test Structure

### Fixtures Available
- `accessibility_test_config`: Configuration for accessibility testing
- `wcag_validator`: WCAG compliance validation utilities
- `color_contrast_analyzer`: Color contrast analysis tools
- `keyboard_navigation_tester`: Keyboard navigation testing utilities
- `screen_reader_validator`: Screen reader compatibility validation
- `mock_axe_core`: Mock axe-core testing engine
- `accessibility_test_data`: Test data for accessibility validation

### Example Test Implementation
```python
import pytest

@pytest.mark.accessibility
@pytest.mark.wcag
def test_wcag_compliance(wcag_validator, accessibility_test_data):
    """Test WCAG 2.1 AA compliance."""
    html_content = accessibility_test_data["valid_html"]
    
    result = wcag_validator.validate_html_structure(html_content)
    
    assert result["compliant"], f"WCAG validation failed: {result['issues']}"
    assert result["errors"] == 0, "Should have no WCAG errors"
```

## Browser Automation Testing

### Playwright Integration
```python
@pytest.mark.asyncio
async def test_keyboard_navigation_with_playwright():
    """Test keyboard navigation using Playwright."""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        await page.goto("http://localhost:3000")
        
        # Test tab navigation
        await page.keyboard.press("Tab")
        focused_element = await page.evaluate("document.activeElement.tagName")
        assert focused_element in ["A", "BUTTON", "INPUT"], "Should focus interactive element"
        
        await browser.close()
```

### axe-core Integration
```python
@pytest.mark.asyncio
async def test_axe_core_validation():
    """Test with axe-core via Playwright."""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        await page.goto("http://localhost:3000")
        
        # Inject axe-core
        await page.add_script_tag(url="https://unpkg.com/axe-core@4.8.0/axe.min.js")
        
        # Run axe analysis
        results = await page.evaluate("""
            () => new Promise((resolve) => {
                axe.run((err, results) => {
                    if (err) throw err;
                    resolve(results);
                });
            })
        """)
        
        assert len(results["violations"]) == 0, f"axe violations found: {results['violations']}"
        
        await browser.close()
```

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Accessibility Tests

on: [push, pull_request]

jobs:
  accessibility:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.12'
        
    - name: Install dependencies
      run: |
        pip install uv
        uv sync --group accessibility
        
    - name: Install Playwright browsers
      run: uv run playwright install
      
    - name: Start test server
      run: |
        uv run python -m http.server 3000 &
        sleep 5
        
    - name: Run accessibility tests
      run: uv run pytest tests/accessibility/ -v --junit-xml=accessibility-results.xml
      
    - name: Upload test results
      uses: actions/upload-artifact@v4
      if: always()
      with:
        name: accessibility-test-results
        path: accessibility-results.xml
```

## Manual Testing Procedures

### Screen Reader Testing
1. **NVDA (Windows)**:
   ```
   - Install NVDA screen reader
   - Navigate to application
   - Use NVDA key + Ctrl + T to read title
   - Use H to navigate headings
   - Use F to navigate forms
   - Use L to navigate links
   ```

2. **VoiceOver (macOS)**:
   ```
   - Enable VoiceOver (Cmd + F5)
   - Use VO + A to read all
   - Use VO + Command + H for headings
   - Use VO + Command + L for links
   - Use VO + Command + J for form controls
   ```

### Keyboard Navigation Testing
1. **Tab Order**: Use Tab key to navigate through all interactive elements
2. **Focus Indicators**: Verify visible focus indicators on all focusable elements
3. **Skip Links**: Test skip link functionality with Tab and Enter
4. **Keyboard Shortcuts**: Test any custom keyboard shortcuts
5. **Modal Focus**: Verify focus trapping in modal dialogs

### Color Contrast Testing
1. **Browser Developer Tools**: Use accessibility panel to check contrast
2. **Online Tools**: Use WebAIM Color Contrast Checker
3. **Automated Tools**: Run axe-core contrast checks
4. **Manual Verification**: Print in grayscale to verify contrast

## Accessibility Checklist

### WCAG 2.1 AA Compliance Checklist

#### Perceivable
- [ ] All images have appropriate alt text
- [ ] Color is not the only way information is conveyed
- [ ] Text has sufficient contrast ratio (4.5:1 normal, 3:1 large)
- [ ] Text can be resized up to 200% without loss of functionality
- [ ] No auto-playing audio

#### Operable
- [ ] All functionality is keyboard accessible
- [ ] No keyboard traps exist
- [ ] Users can pause, stop, or hide moving content
- [ ] No content flashes more than 3 times per second
- [ ] Skip links are provided for repetitive content
- [ ] Page titles are descriptive and unique
- [ ] Focus order is logical
- [ ] Focus indicators are visible

#### Understandable
- [ ] Page language is identified
- [ ] Language changes are identified
- [ ] Navigation is consistent across pages
- [ ] Elements with same functionality are consistently identified
- [ ] Form errors are clearly identified
- [ ] Labels and instructions are provided for form inputs
- [ ] Error suggestions are provided when possible

#### Robust
- [ ] Markup validates and uses proper semantics
- [ ] ARIA is used appropriately
- [ ] Content works with assistive technologies

## Reporting and Documentation

### Test Report Generation
The accessibility testing framework automatically generates comprehensive reports including:

- Overall accessibility score
- WCAG compliance status
- Tool-specific results (axe, pa11y, Lighthouse)
- Priority fixes list
- Recommendations for improvement

### Report Structure
```json
{
  "timestamp": "2024-01-01T00:00:00Z",
  "page_url": "http://test.example.com",
  "overall_score": 0.85,
  "wcag_compliance": {
    "level_a": {"compliant": true, "violations": 0},
    "level_aa": {"compliant": false, "violations": 3},
    "level_aaa": {"compliant": false, "violations": 8}
  },
  "testing_tools": {
    "axe_core": {"violations": 2, "passes": 15},
    "pa11y": {"errors": 1, "warnings": 2},
    "lighthouse": {"accessibility_score": 0.87}
  },
  "priority_fixes": [
    {
      "priority": "critical",
      "issue": "Images missing alt text",
      "affected_elements": 3
    }
  ],
  "recommendations": [
    "Fix all critical accessibility violations",
    "Improve color contrast ratios",
    "Add proper form labels"
  ]
}
```

## Best Practices

### Development Guidelines
1. **Semantic HTML First**: Use proper HTML elements before adding ARIA
2. **Progressive Enhancement**: Ensure basic functionality without JavaScript
3. **Test Early and Often**: Include accessibility testing in development workflow
4. **Real User Testing**: Test with actual screen reader users when possible
5. **Automated Testing**: Use automated tools but don't rely on them exclusively

### Testing Guidelines
1. **Test with Multiple Tools**: Use axe-core, pa11y, and Lighthouse together
2. **Manual Testing**: Always include manual keyboard and screen reader testing
3. **Real Devices**: Test on actual devices and assistive technologies
4. **User Scenarios**: Test complete user workflows, not just individual components
5. **Regular Audits**: Conduct accessibility audits at regular intervals

### Documentation Guidelines
1. **Document Exceptions**: Clearly document any accessibility exceptions with justification
2. **Provide Alternatives**: Document alternative methods for accessing functionality
3. **Update Guidelines**: Keep accessibility guidelines current with WCAG updates
4. **Training Materials**: Maintain accessibility training materials for the team

## Resources

### WCAG Guidelines
- [WCAG 2.1 Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)
- [WCAG 2.1 Understanding Document](https://www.w3.org/WAI/WCAG21/Understanding/)

### Testing Tools
- [axe-core Documentation](https://github.com/dequelabs/axe-core)
- [pa11y Documentation](https://github.com/pa11y/pa11y)
- [Lighthouse Accessibility Audit](https://developers.google.com/web/tools/lighthouse/audits/accessibility)

### Screen Readers
- [NVDA Download](https://www.nvaccess.org/download/)
- [VoiceOver User Guide](https://support.apple.com/guide/voiceover/)
- [JAWS Information](https://www.freedomscientific.com/products/software/jaws/)

### Color Contrast Tools
- [WebAIM Contrast Checker](https://webaim.org/resources/contrastchecker/)
- [Colour Contrast Analyser](https://www.tpgi.com/color-contrast-checker/)

## Support

For questions about accessibility testing or implementation:

1. Check this guide and existing test examples
2. Review WCAG 2.1 guidelines for specific requirements
3. Run automated tools for initial validation
4. Conduct manual testing to verify automation results
5. Consider user testing with individuals who use assistive technology

Remember: Accessibility is not just about compliance—it's about creating inclusive experiences for all users.