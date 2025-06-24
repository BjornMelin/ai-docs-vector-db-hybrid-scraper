# Visual Regression Testing Framework

This directory contains comprehensive visual regression testing for the AI Documentation Vector DB Hybrid Scraper, ensuring UI consistency and preventing visual regressions across different browsers, devices, and screen sizes.

## Framework Overview

The visual regression testing framework provides:

- **Automated screenshot capture** with pixel-perfect comparison
- **Cross-browser visual validation** across Chrome, Firefox, Safari, and Edge
- **Responsive design testing** for multiple screen sizes and devices
- **Component-level visual testing** for UI elements and interactions
- **Visual diff analysis** with automated change detection and reporting

## Directory Structure

- **screenshots/**: Screenshot capture and management system
- **baseline/**: Baseline image storage and version management
- **comparison/**: Visual difference detection and analysis
- **ui_components/**: Component-level visual testing
- **responsive/**: Responsive design visual validation

## Core Visual Testing Capabilities

### Screenshot Management

```python
# Automated screenshot capture
@pytest.mark.visual
def test_screenshot_capture():
    """Capture and validate page screenshots."""
    pass
```

**Key Features:**
- High-resolution screenshot capture
- Element-specific screenshot targeting
- Multiple viewport size support
- Cross-browser screenshot consistency
- Screenshot metadata and versioning

### Baseline Management

```python
# Baseline image management
@pytest.mark.visual
@pytest.mark.baseline
def test_baseline_update():
    """Update visual baseline images."""
    pass
```

**Key Features:**
- Automated baseline image storage
- Version control integration for baselines
- Baseline approval workflow
- Multi-environment baseline management
- Baseline image optimization and compression

### Visual Comparison

```python
# Visual difference detection
@pytest.mark.visual
@pytest.mark.comparison
def test_visual_diff():
    """Detect visual differences from baseline."""
    pass
```

**Key Features:**
- Pixel-level difference detection
- Configurable comparison sensitivity
- Difference highlighting and annotation
- False positive reduction algorithms
- Statistical analysis of visual changes

### Component Testing

```python
# UI component visual validation
@pytest.mark.visual
@pytest.mark.component
def test_component_visual():
    """Test individual UI component visuals."""
    pass
```

**Key Features:**
- Isolated component screenshot capture
- Component state variation testing
- Interactive element visual validation
- Component consistency across pages
- Design system compliance checking

### Responsive Testing

```python
# Responsive design visual validation
@pytest.mark.visual
@pytest.mark.responsive
def test_responsive_design():
    """Test responsive design across devices."""
    pass
```

**Key Features:**
- Multiple device simulation
- Breakpoint visual validation
- Mobile and tablet layout testing
- Orientation change testing
- Dynamic content responsive behavior

## Usage Commands

### Quick Start

```bash
# Run all visual regression tests
uv run pytest tests/visual_regression/ -v

# Run specific visual testing category
uv run pytest tests/visual_regression/screenshots/ -v
uv run pytest tests/visual_regression/responsive/ -v
uv run pytest tests/visual_regression/ui_components/ -v

# Run with visual markers
uv run pytest -m "visual" -v
```

### Screenshot Operations

```bash
# Capture new screenshots
uv run pytest tests/visual_regression/ --visual-capture

# Update baseline images
uv run pytest tests/visual_regression/ --visual-update-baseline

# Generate visual diff reports
uv run pytest tests/visual_regression/ --visual-diff-report

# Run with specific browser
uv run pytest tests/visual_regression/ --browser=chrome
uv run pytest tests/visual_regression/ --browser=firefox
```

### Cross-Browser Testing

```bash
# Run visual tests across multiple browsers
uv run pytest tests/visual_regression/ --browser=chrome,firefox,safari

# Test specific screen resolutions
uv run pytest tests/visual_regression/ --resolution=1920x1080,1366x768,375x667

# Mobile and tablet testing
uv run pytest tests/visual_regression/ --device=mobile,tablet,desktop
```

### CI/CD Integration

```bash
# Visual regression tests for CI
uv run pytest tests/visual_regression/ --visual-ci-mode --tb=short

# Generate visual test report
uv run pytest tests/visual_regression/ --visual-report=html --visual-report-dir=reports/
```

## Visual Testing Strategies

### Page-Level Testing

Test complete page layouts and overall visual consistency:

```python
@pytest.mark.visual
def test_homepage_visual():
    """Test homepage visual consistency."""
    # Navigate to homepage
    # Capture full page screenshot
    # Compare with baseline
    # Report any differences
    pass
```

### Component-Level Testing

Test individual UI components in isolation:

```python
@pytest.mark.visual
@pytest.mark.component
def test_search_component_visual():
    """Test search component visual states."""
    # Test normal state
    # Test focused state
    # Test with results
    # Test error state
    pass
```

### Responsive Design Testing

Validate layout across different screen sizes:

```python
@pytest.mark.visual
@pytest.mark.responsive
@pytest.mark.parametrize("viewport", [
    (320, 568),   # Mobile
    (768, 1024),  # Tablet
    (1920, 1080), # Desktop
])
def test_responsive_layout(viewport):
    """Test responsive layout at different sizes."""
    pass
```

### Interactive Element Testing

Test visual states of interactive elements:

```python
@pytest.mark.visual
@pytest.mark.interaction
def test_button_states():
    """Test button visual states."""
    # Test default state
    # Test hover state
    # Test active state
    # Test disabled state
    pass
```

## Visual Testing Configuration

### Browser Configuration

```yaml
# visual_config.yml
browsers:
  - chrome:
      headless: true
      window_size: [1920, 1080]
  - firefox:
      headless: true
      window_size: [1920, 1080]
  - safari:
      headless: false  # Safari doesn't support headless
      window_size: [1920, 1080]
```

### Comparison Settings

```yaml
# Comparison sensitivity settings
comparison:
  threshold: 0.1        # 10% difference threshold
  ignore_colors: false  # Consider color differences
  ignore_fonts: false   # Consider font differences
  ignore_size: false    # Consider size differences
  pixel_threshold: 5    # Pixel-level threshold
```

### Device Profiles

```yaml
# Device simulation profiles
devices:
  mobile:
    - iPhone_12: [390, 844]
    - Samsung_Galaxy: [360, 740]
  tablet:
    - iPad: [768, 1024]
    - iPad_Pro: [1024, 1366]
  desktop:
    - HD: [1366, 768]
    - Full_HD: [1920, 1080]
    - 4K: [3840, 2160]
```

## Integration Points

### CI/CD Pipeline Integration

```yaml
# GitHub Actions visual testing
name: Visual Regression Tests
on: [pull_request]

jobs:
  visual-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run visual regression tests
        run: |
          uv run pytest tests/visual_regression/ \
            --visual-ci-mode \
            --visual-report=junit \
            --visual-baseline-remote=origin/main
```

### Design System Integration

- Component library visual validation
- Design token consistency checking
- Brand guideline compliance testing
- Accessibility visual standards validation

### Development Workflow Integration

- Pre-commit visual checks for changed components
- Pull request visual diff reports
- Automated visual review assignments
- Visual change approval workflows

## Visual Testing Tools

### Browser Automation
- **Playwright**: Cross-browser automation and screenshot capture
- **Selenium**: Legacy browser support and automation
- **Puppeteer**: Chrome-specific automation and advanced features

### Image Processing
- **Pillow (PIL)**: Image manipulation and comparison
- **ImageIO**: Advanced image processing and analysis
- **OpenCV**: Computer vision-based visual analysis
- **scikit-image**: Scientific image processing

### Reporting and Analysis
- **pytest-html**: HTML test reporting with visual diffs
- **Allure**: Advanced test reporting with visual evidence
- **Slack/Teams integration**: Automated visual regression alerts
- **Custom dashboards**: Visual testing metrics and trends

## Best Practices

### Baseline Management
- Establish clear baseline approval processes
- Version control baseline images with code changes
- Regular baseline maintenance and cleanup
- Environment-specific baseline management
- Automated baseline validation

### Test Stability
- Use stable test data and content
- Handle dynamic content appropriately
- Implement proper wait strategies
- Account for animation and transitions
- Minimize test flakiness with retry mechanisms

### Performance Optimization
- Optimize screenshot capture performance
- Use parallel execution for faster testing
- Implement intelligent test selection
- Cache baseline images efficiently
- Minimize network dependencies

### Maintenance and Monitoring
- Regular visual test maintenance
- Monitor visual test execution times
- Track visual regression detection rates
- Analyze false positive patterns
- Continuous improvement of visual testing accuracy

This visual regression testing framework ensures consistent visual quality and prevents UI regressions across the AI Documentation Vector DB Hybrid Scraper application.