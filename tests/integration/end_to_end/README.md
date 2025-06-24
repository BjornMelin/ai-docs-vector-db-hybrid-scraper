# End-to-End User Journey Testing Infrastructure

This directory contains comprehensive end-to-end testing infrastructure that validates complete user workflows across the entire AI Documentation Vector DB Hybrid Scraper system.

## Overview

The E2E testing infrastructure implements **Groups 1, 2, and 3** requirements plus comprehensive user journey validation:

- **Complete User Journey Tests**: Full workflows from document ingestion to search results
- **Browser Automation**: Real browser interactions using Playwright
- **API Workflow Validation**: Complete API client interactions and endpoint testing
- **System Integration**: Cross-component workflow validation
- **Real-World Scenarios**: Realistic user scenarios with actual data flows

## Directory Structure

```
end_to_end/
├── user_journeys/           # Core user journey tests
│   ├── conftest.py         # Journey testing fixtures and utilities
│   └── test_complete_user_journeys.py  # Main user journey test suite
├── browser_automation/      # Browser-based testing
│   └── test_browser_user_journeys.py   # Playwright automation tests
├── api_flows/              # API workflow testing
│   └── test_api_workflow_validation.py # API client journey tests
├── workflow_testing/       # System workflow tests
│   └── test_system_workflows.py       # Multi-component workflows
├── system_integration/     # Full system integration
│   └── test_end_to_end_integration.py # Complete system scenarios
└── README.md              # This file
```

## Test Categories

### 1. User Journey Tests (`user_journeys/`)

**Complete workflow validation from start to finish:**

- **Document Processing Journey**: URL crawling → content extraction → embedding generation → vector storage → search validation
- **Search and Discovery Journey**: Query processing → vector search → result ranking → response formatting  
- **Project Management Journey**: Project creation → configuration → document addition → analytics
- **API Client Journey**: Authentication → endpoint discovery → request/response validation → error handling
- **Administrative Journey**: System monitoring → performance analysis → troubleshooting workflows

**Key Features:**
- Step-by-step workflow execution with dependency tracking
- Context passing between workflow steps
- Retry logic and error recovery
- Performance metrics collection
- Data flow validation

### 2. Browser Automation Tests (`browser_automation/`)

**Real browser interactions using Playwright:**

- **Documentation Discovery**: Navigate sites → extract content → validate metadata
- **Multi-Page Crawling**: Sequential page processing → content extraction → screenshot capture
- **Form Interactions**: Fill forms → submit data → validate responses
- **Performance Monitoring**: Page load times → resource usage → response metrics
- **Error Handling**: Network failures → timeout recovery → graceful degradation

**Key Features:**
- Cross-platform browser support (Chromium, Firefox, WebKit)
- Headless and headed browser modes
- Screenshot capture for validation
- Performance monitoring integration
- Mobile viewport simulation

### 3. API Workflow Tests (`api_flows/`)

**Complete API client interactions:**

- **Authentication Workflow**: Login → token management → authenticated requests
- **Document Management**: Project creation → collection setup → document addition → validation
- **Search Workflow**: Query processing → result retrieval → quality validation
- **Error Handling**: Error injection → recovery testing → graceful degradation
- **Performance Testing**: Load simulation → throughput measurement → response time validation

**Key Features:**
- Mock API client with realistic responses
- Request/response validation
- Error scenario simulation
- Performance benchmarking
- Rate limiting testing

### 4. System Workflows (`workflow_testing/`)

**Multi-component system workflows:**

- **Complete Document Ingestion**: Cross-service data flow validation
- **Search and Retrieval**: End-to-end search pipeline testing
- **Multi-Tenant**: Isolated data processing validation
- **Failure Recovery**: Component failure simulation and recovery
- **High Throughput**: Performance under load conditions
- **Data Consistency**: Cross-component data integrity validation

**Key Features:**
- Component health monitoring
- Data flow validation
- Performance tracking
- Error injection and recovery
- Scalability testing

### 5. System Integration (`system_integration/`)

**Complete system integration scenarios:**

- **Real-World Documentation**: Realistic document processing workflows
- **Multi-User Concurrent**: Concurrent usage simulation
- **Data Consistency**: System-wide data integrity validation
- **Long-Running Stability**: Extended operation testing
- **Error Recovery**: System resilience validation

**Key Features:**
- Comprehensive scenario execution
- Multi-phase testing workflow
- Performance and stability validation
- Real-world data simulation
- Integration metrics collection

## Usage

### Running Individual Test Categories

```bash
# User journey tests
uv run pytest tests/integration/end_to_end/user_journeys/ -v

# Browser automation tests (requires Playwright)
uv run pytest tests/integration/end_to_end/browser_automation/ -v -m browser

# API workflow tests
uv run pytest tests/integration/end_to_end/api_flows/ -v

# System workflow tests
uv run pytest tests/integration/end_to_end/workflow_testing/ -v

# Complete system integration
uv run pytest tests/integration/end_to_end/system_integration/ -v
```

### Running All E2E Tests

```bash
# All end-to-end tests
uv run pytest tests/integration/end_to_end/ -v

# E2E tests with performance markers
uv run pytest tests/integration/end_to_end/ -v -m \"e2e and performance\"

# E2E tests excluding slow tests
uv run pytest tests/integration/end_to_end/ -v -m \"e2e and not slow\"
```

### Test Markers

- `@pytest.mark.e2e` - End-to-end tests
- `@pytest.mark.integration` - Integration tests  
- `@pytest.mark.browser` - Browser automation tests
- `@pytest.mark.performance` - Performance tests
- `@pytest.mark.slow` - Long-running tests

## Key Features

### Journey Executor Framework

The `JourneyExecutor` provides a flexible framework for defining and executing complex user workflows:

```python
# Define a custom journey
journey = UserJourney(
    name="custom_workflow",
    description="Custom user workflow",
    steps=[
        JourneyStep(
            name="step_1",
            action="crawl_url", 
            params={"url": "https://example.com"},
            timeout_seconds=30.0,
        ),
        JourneyStep(
            name="step_2",
            action="process_document",
            params={"content": "${context.content}"},
            dependencies=["content"],
        ),
    ],
    success_criteria={"min_success_rate": 0.9, "max_errors": 1},
)

# Execute the journey
result = await journey_executor.execute_journey(journey)
```

### Browser Automation

Real browser testing with Playwright integration:

```python
# Browser automation with real interactions
async def test_documentation_workflow(page):
    await page.goto("https://docs.example.com")
    content = await page.content()
    
    # Extract and validate content
    metadata = await page.evaluate("() => ({ title: document.title })")
    assert metadata["title"], "Page should have title"
```

### API Workflow Validation

Complete API client testing with mock services:

```python
# API workflow testing
async def test_api_workflow(mock_api_client):
    # Authentication
    auth_response = await mock_api_client.request("POST", "/auth/login")
    
    # Authenticated operations
    projects = await mock_api_client.request(
        "GET", "/projects",
        headers={"Authorization": f"Bearer {auth_response['token']}"}
    )
```

### System Integration

Multi-component workflow orchestration:

```python
# System workflow orchestration
workflow_result = await orchestrator.execute_workflow(
    workflow_name="document_processing",
    components=["crawler", "processor", "embeddings", "storage"],
    workflow_steps=processing_steps,
    success_criteria={"min_component_health_rate": 0.9}
)
```

## Data Flow Validation

The testing infrastructure validates data flow across all system components:

1. **Ingestion → Processing**: Documents are correctly processed and chunked
2. **Processing → Embeddings**: Chunks are converted to vector embeddings  
3. **Embeddings → Storage**: Vectors are stored in the vector database
4. **Storage → Search**: Stored vectors are searchable and retrievable
5. **Search → Results**: Search results are properly ranked and formatted

## Performance Validation

Comprehensive performance testing includes:

- **Response Time Monitoring**: Track API and search response times
- **Throughput Measurement**: Measure requests/documents processed per second
- **Concurrency Testing**: Validate performance under concurrent load
- **Resource Usage**: Monitor memory and CPU usage during operations
- **Scalability Testing**: Test system behavior as load increases

## Error Recovery Testing

Robust error handling validation:

- **Network Failures**: Timeout and connectivity issues
- **Service Unavailability**: Component failure simulation
- **Rate Limiting**: API rate limit handling
- **Data Corruption**: Invalid data handling and recovery
- **Resource Exhaustion**: Memory and disk space limitations

## Integration with Existing Infrastructure

The E2E testing leverages existing test infrastructure from Groups 1-3:

- **Load Testing**: Uses load testing fixtures for realistic scenario simulation
- **Security Testing**: Integrates authentication and authorization validation
- **Contract Testing**: Validates API contracts during workflow execution
- **Chaos Engineering**: Incorporates failure injection for resilience testing
- **Accessibility**: Validates UI accessibility during browser automation

## Test Data Management

- **Realistic Test Data**: Uses representative URLs and content
- **Data Isolation**: Ensures test data doesn't interfere between tests
- **Cleanup Automation**: Automatic cleanup of test artifacts
- **State Management**: Maintains test state across complex workflows
- **Reproducibility**: Consistent test conditions for reliable results

## Monitoring and Observability

- **Journey Metrics**: Detailed metrics for each user journey
- **Performance Tracking**: Response times, throughput, and resource usage
- **Error Logging**: Comprehensive error capture and analysis
- **Health Monitoring**: System health tracking during test execution
- **Artifact Storage**: Test artifacts stored for post-test analysis

## Best Practices

### Writing User Journey Tests

1. **Define Clear Steps**: Each step should have a specific, testable action
2. **Use Dependencies**: Declare data dependencies between steps
3. **Set Realistic Timeouts**: Allow sufficient time for each operation
4. **Validate Results**: Include meaningful assertions for each step
5. **Handle Errors**: Plan for failure scenarios and recovery

### Browser Automation

1. **Use Stable Selectors**: Prefer data attributes over fragile CSS selectors
2. **Wait for Elements**: Use explicit waits for dynamic content
3. **Capture Screenshots**: Include screenshots for debugging failures
4. **Test Across Browsers**: Validate functionality across browser types
5. **Handle Async Operations**: Properly wait for asynchronous operations

### API Testing

1. **Mock Realistically**: Create mocks that behave like real services
2. **Test Error Conditions**: Include negative test scenarios
3. **Validate Responses**: Check both success and error response formats
4. **Test Authentication**: Include authentication workflows
5. **Monitor Performance**: Track API response times and throughput

## Troubleshooting

### Common Issues

1. **Browser Tests Failing in CI**: 
   - Ensure Playwright browsers are installed
   - Use headless mode in CI environments
   - Check browser sandbox settings

2. **Journey Step Timeouts**:
   - Increase timeout values for slow operations
   - Check network connectivity in test environment
   - Validate mock service response times

3. **Data Flow Validation Failures**:
   - Check context variable passing between steps
   - Validate dependency declarations
   - Ensure mock services return expected data formats

4. **Performance Test Failures**:
   - Adjust performance thresholds for test environment
   - Account for CI environment variability
   - Check resource availability during tests

### Debug Mode

Enable debug logging for detailed test execution information:

```bash
# Enable debug logging
PYTEST_CURRENT_TEST=1 uv run pytest tests/integration/end_to_end/ -v -s --log-level=DEBUG
```

## Future Enhancements

The E2E testing infrastructure is designed for extensibility:

- **Custom Journey Actions**: Add domain-specific workflow actions
- **Additional Browser Engines**: Support for additional browser automation tools
- **Real Service Integration**: Connect to actual services for integration testing
- **Advanced Performance Analysis**: Detailed performance profiling and analysis
- **Visual Regression Testing**: Screenshot comparison for UI validation
- **Load Testing Integration**: Enhanced load testing during user journeys

## Contributing

When adding new E2E tests:

1. Follow the existing patterns for journey definition
2. Include comprehensive error handling and recovery
3. Add performance validation where appropriate
4. Include realistic test data and scenarios
5. Document any new fixtures or utilities
6. Update this README with new test categories or features

The E2E testing infrastructure provides comprehensive validation of the entire AI Documentation Vector DB Hybrid Scraper system, ensuring reliable operation across all user workflows and system components.