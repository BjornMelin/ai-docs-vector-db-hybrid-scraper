# Accessibility Testing Suite

This directory contains comprehensive accessibility testing for the AI Documentation Vector DB Hybrid Scraper web interfaces.

## Directory Structure

- **a11y/**: General accessibility testing and validation
- **wcag/**: Web Content Accessibility Guidelines (WCAG) compliance testing
- **screen_reader/**: Screen reader compatibility testing
- **keyboard_navigation/**: Keyboard-only navigation testing
- **color_contrast/**: Color contrast and visual accessibility testing
- **aria/**: ARIA attributes and semantic HTML testing

## Running Accessibility Tests

```bash
# Run all accessibility tests
uv run pytest tests/accessibility/ -v

# Run specific category
uv run pytest tests/accessibility/wcag/ -v

# Run with accessibility markers
uv run pytest -m accessibility -v
```

## Test Categories

### WCAG Compliance (wcag/)
- WCAG 2.1 AA compliance testing
- WCAG 2.2 compliance validation
- Accessibility guideline adherence

### Screen Reader Testing (screen_reader/)
- NVDA compatibility testing
- JAWS compatibility testing
- VoiceOver compatibility testing
- Screen reader text reading validation

### Keyboard Navigation (keyboard_navigation/)
- Tab order validation
- Focus management testing
- Keyboard shortcut functionality
- Skip link testing

### Color Contrast (color_contrast/)
- Color contrast ratio validation
- Color blindness simulation
- High contrast mode testing

### ARIA Testing (aria/)
- ARIA label validation
- ARIA role testing
- ARIA state management
- Semantic HTML validation

## Tools and Frameworks

- **axe-core**: Automated accessibility testing
- **playwright**: Browser automation for accessibility testing
- **pytest-accessibility**: Accessibility-specific pytest plugins
- **axe-playwright**: Playwright integration for axe-core