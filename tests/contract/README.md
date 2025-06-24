# Contract Testing Suite

This directory contains contract testing for API interfaces and service integrations.

## Directory Structure

- **api_contracts/**: API contract validation and testing
- **schema_validation/**: JSON schema and data structure validation
- **pact/**: Consumer-driven contract testing with Pact
- **openapi/**: OpenAPI specification testing and validation
- **consumer_driven/**: Consumer-driven contract testing scenarios

## Running Contract Tests

```bash
# Run all contract tests
uv run pytest tests/contract/ -v

# Run API contract tests
uv run pytest tests/contract/api_contracts/ -v

# Run with contract markers
uv run pytest -m contract -v
```

## Test Categories

### API Contract Testing (api_contracts/)
- REST API contract validation
- Request/response structure validation
- API versioning compatibility testing
- Backward compatibility validation

### Schema Validation (schema_validation/)
- JSON schema validation
- Pydantic model validation
- Database schema validation
- Configuration schema testing

### Pact Testing (pact/)
- Consumer contract generation
- Provider contract verification
- Contract evolution testing
- Cross-service compatibility

### OpenAPI Testing (openapi/)
- OpenAPI specification validation
- API documentation accuracy
- Request/response example validation
- Swagger/OpenAPI compliance

### Consumer-Driven Testing (consumer_driven/)
- Consumer expectation validation
- Service integration testing
- API consumer contract verification
- Interface compatibility testing

## Tools and Frameworks

- **schemathesis**: Property-based API testing using OpenAPI specs
- **openapi-spec-validator**: OpenAPI specification validation
- **jsonschema**: JSON schema validation
- **pact-python**: Python Pact implementation for consumer-driven contracts
- **httpx**: HTTP client for contract testing
- **responses**: HTTP request mocking for testing
- **factory-boy**: Test data generation
- **faker**: Fake data generation for testing

## Installation

Install contract testing dependencies:

```bash
# Install contract testing tools
uv add --group contract schemathesis openapi-spec-validator jsonschema pact-python responses httpx factory-boy faker

# Or install with the contract group
uv sync --group contract
```

## Advanced Usage

### Property-Based Testing with Schemathesis

```python
import schemathesis

# Load schema from FastAPI app
schema = schemathesis.from_asgi("/openapi.json", app)

@schema.parametrize()
def test_api_contract(case):
    response = case.call_asgi()
    case.validate_response(response)
```

### Pact Consumer-Driven Contracts

```python
from pact import Consumer, Provider

pact = Consumer('ai-docs-consumer').has_pact_with(Provider('ai-docs-provider'))

pact.given('documents exist').upon_receiving('search request').with_request(
    method='GET',
    path='/search',
    query={'q': 'test'}
).will_respond_with(200, body={'results': []})
```

### Breaking Change Detection

The framework automatically detects:
- New required fields in requests/responses
- Removed fields that were previously required
- Changed field types
- Modified validation rules
- API parameter changes

### Performance Testing

Contract validation performance is monitored:
- Validation throughput (operations/second)
- Memory usage during validation
- Concurrent validation capabilities

## CI/CD Integration

Add to your CI pipeline:

```yaml
- name: Run Contract Tests
  run: |
    uv run pytest tests/contract/ -v --tb=short
    uv run pytest tests/contract/ -m "contract and not slow" --maxfail=5
```

## Reporting

Contract test results are automatically saved to:
- `tests/contract/reports/contract-test-report.json`
- JUnit XML format for CI integration
- HTML reports for detailed analysis