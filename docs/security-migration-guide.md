# Security Module Migration Guide

## Overview

In V1, we've transitioned from the legacy security module (`src/security.py`) to a new minimalistic ML security framework (`src/security/ml_security.py`). This guide helps you migrate existing code.

## Key Changes

### 1. Module Structure
- **Old**: Single file `src/security.py`
- **New**: Module directory `src/security/` with specialized files

### 2. Class Renaming
| Old Class | New Class | Purpose |
|-----------|-----------|---------|
| `SecurityValidator` | `MLSecurityValidator` | Input validation for ML systems |
| `SecurityError` | (removed) | Use standard exceptions |
| `APIKeyValidator` | (removed) | Use auth middleware instead |

### 3. Import Changes

**Before:**
```python
from src.security import SecurityValidator
from src.security import SecurityError
from src.security import APIKeyValidator
```

**After:**
```python
from src.security import MLSecurityValidator as SecurityValidator
# SecurityError removed - use ValueError or custom exceptions
# APIKeyValidator removed - use FastAPI dependencies
```

## Migration Steps

### Step 1: Update Imports

Search and replace in your codebase:
```bash
# Find all SecurityValidator imports
grep -r "from.*security.*import.*SecurityValidator" src/

# Replace with new import
from src.security import MLSecurityValidator as SecurityValidator
```

### Step 2: Update Error Handling

**Old pattern:**
```python
try:
    validator.validate_input(data)
except SecurityError as e:
    handle_error(e)
```

**New pattern:**
```python
result = validator.validate_input(data)
if not result.is_valid:
    handle_error(result.error_message)
```

### Step 3: API Key Validation

**Old pattern:**
```python
api_validator = APIKeyValidator(config)
if not api_validator.validate(api_key):
    raise Unauthorized()
```

**New pattern (using FastAPI dependencies):**
```python
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if not is_valid_token(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid authentication")
    return credentials.credentials
```

## New Features in ML Security Module

### 1. Input Validation
```python
from src.security import MLSecurityValidator

validator = MLSecurityValidator()
result = validator.validate_text_input(
    text="User input here",
    max_length=1000,
    check_injection=True
)

if not result.is_valid:
    print(f"Validation failed: {result.error_message}")
```

### 2. Rate Limiting
```python
from src.security import SimpleRateLimiter

rate_limiter = SimpleRateLimiter(
    max_requests=100,
    window_seconds=60
)

if not rate_limiter.check_rate_limit(user_id):
    raise HTTPException(status_code=429, detail="Rate limit exceeded")
```

### 3. Security Configuration
```python
from src.security import MinimalMLSecurityConfig

config = MinimalMLSecurityConfig(
    enable_input_validation=True,
    enable_output_filtering=True,
    max_input_length=10000,
    rate_limit_requests=100,
    rate_limit_window=60
)
```

## Testing Your Migration

1. **Run existing tests**: Ensure all tests pass after migration
2. **Check imports**: Verify no old imports remain
   ```bash
   grep -r "from src.security import" src/ | grep -v "MLSecurityValidator"
   ```
3. **Validate functionality**: Test security features in your application

## Rollback Plan

If you need to temporarily rollback:
1. Keep the old `src/security.py` file as `src/security_legacy.py`
2. Update imports to use the legacy module
3. Plan migration in phases

## Support

For questions or issues during migration:
1. Check the [ML Security Documentation](./ml-security-framework.md)
2. Review test examples in `tests/unit/security/`
3. Create an issue in the project repository

## Timeline

- **V1.0**: Both modules available, new module recommended
- **V1.1**: Deprecation warnings added to old module
- **V2.0**: Old module removed completely

---

*Migration guide version: 1.0*
*Last updated: 2025-06-14*