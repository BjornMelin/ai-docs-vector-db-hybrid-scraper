# Security Testing Suite

This directory contains comprehensive security testing for the AI Documentation Vector DB Hybrid Scraper.

## Directory Structure

- **vulnerability/**: Automated vulnerability scanning and assessment tests
- **penetration/**: Penetration testing scenarios and attack simulations
- **compliance/**: Security compliance testing (OWASP, NIST, etc.)
- **authentication/**: Authentication mechanism testing
- **authorization/**: Authorization and access control testing
- **input_validation/**: Input sanitization and validation testing
- **encryption/**: Encryption and data protection testing

## Running Security Tests

```bash
# Run all security tests
uv run pytest tests/security/ -v

# Run specific category
uv run pytest tests/security/vulnerability/ -v

# Run with security markers
uv run pytest -m security -v
```

## Test Categories

### Vulnerability Testing
- Dependency vulnerability scanning
- Code vulnerability analysis
- Configuration security assessment

### Penetration Testing
- SQL injection testing
- XSS vulnerability testing
- Authentication bypass attempts
- API endpoint security testing

### Compliance Testing
- OWASP Top 10 compliance
- Data protection regulation compliance
- Security policy validation

### Authentication & Authorization
- JWT token validation
- Session management testing
- Role-based access control testing
- API key security testing

### Input Validation
- SQL injection prevention
- XSS prevention
- Path traversal prevention
- File upload security

### Encryption Testing
- Data at rest encryption
- Data in transit encryption
- Key management validation