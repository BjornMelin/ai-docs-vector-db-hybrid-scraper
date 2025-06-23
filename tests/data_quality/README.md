# Data Quality Testing Framework

This directory contains comprehensive data quality testing for the AI Documentation Vector DB Hybrid Scraper, ensuring data integrity, consistency, and reliability across all data processing pipelines.

## Framework Overview

The data quality testing framework provides:

- **Data integrity validation** with comprehensive schema and constraint checking
- **Data consistency verification** across distributed systems and data stores
- **Migration testing** for database schema changes and data transformations  
- **Transformation validation** for data processing pipelines
- **Quality metrics collection** with automated reporting and alerting

## Directory Structure

- **integrity/**: Data integrity validation and constraint checking
- **validation/**: Input data validation and sanitization testing
- **migration/**: Database migration and schema change testing
- **consistency/**: Cross-system data consistency verification
- **transformation/**: Data transformation pipeline testing

## Core Capabilities

### Data Integrity Testing

- **Schema validation** for document structures and metadata
- **Constraint checking** for foreign keys, unique constraints, and data relationships
- **Referential integrity** across vector databases and relational stores
- **Data type validation** and format checking
- **Null value handling** and missing data detection

### Data Consistency Testing

- **Cross-database consistency** between vector DB and metadata stores
- **Cache coherence validation** across multiple cache layers
- **Eventual consistency testing** for distributed systems
- **Data synchronization verification** between services
- **Conflict resolution testing** for concurrent data updates

### Migration Testing

- **Schema migration validation** with rollback testing
- **Data migration integrity** with before/after comparisons
- **Migration performance testing** for large datasets
- **Zero-downtime migration** verification
- **Migration rollback and recovery** testing

### Transformation Testing

- **Document processing pipeline** validation
- **Embedding generation consistency** across different models
- **Data enrichment accuracy** for metadata extraction
- **Chunking strategy validation** for document segmentation
- **Format conversion testing** between different data representations

## Usage Commands

### Quick Start

```bash
# Run all data quality tests
uv run pytest tests/data_quality/ -v

# Run specific data quality category
uv run pytest tests/data_quality/integrity/ -v
uv run pytest tests/data_quality/consistency/ -v
uv run pytest tests/data_quality/migration/ -v

# Run with data quality markers
uv run pytest -m "data_quality" -v
```

### Advanced Testing

```bash
# Run data integrity validation
uv run pytest tests/data_quality/integrity/ -v --tb=short

# Test data consistency across services
uv run pytest tests/data_quality/consistency/ -v --maxfail=5

# Validate data transformations
uv run pytest tests/data_quality/transformation/ -v --capture=no

# Run migration tests with rollback
uv run pytest tests/data_quality/migration/ -v --setup-show
```

### CI/CD Integration

```bash
# Fast data quality checks for CI
uv run pytest tests/data_quality/ -m "fast and not slow" --maxfail=3

# Full data quality validation
uv run pytest tests/data_quality/ --tb=line --durations=10
```

## Test Categories

### 1. Data Integrity Tests
- Document schema validation
- Metadata consistency checking
- Embedding vector validation
- Database constraint verification
- Data type and format validation

### 2. Data Consistency Tests
- Vector DB to metadata store consistency
- Cache synchronization validation
- Cross-service data coherence
- Distributed system consistency
- Conflict resolution verification

### 3. Migration Tests
- Schema change validation
- Data migration integrity
- Performance impact assessment
- Rollback mechanism testing
- Migration dependency validation

### 4. Transformation Tests
- Document processing accuracy
- Embedding generation consistency
- Metadata extraction validation
- Data enrichment verification
- Pipeline output validation

### 5. Data Validation Tests
- Input sanitization testing
- Data format validation
- Business rule enforcement
- Data quality metrics calculation
- Anomaly detection validation

## Quality Metrics

### Data Quality KPIs

- **Completeness**: Percentage of non-null required fields
- **Accuracy**: Correctness of data transformations
- **Consistency**: Agreement across data stores
- **Timeliness**: Data freshness and update lag
- **Validity**: Conformance to defined schemas and constraints

### Automated Reporting

- **Quality dashboards** with real-time metrics
- **Alerting** for data quality threshold breaches
- **Trend analysis** for data quality over time
- **Root cause analysis** for quality issues
- **Remediation tracking** for identified problems

## Integration Points

### Database Systems
- **Qdrant vector database** integrity validation
- **PostgreSQL metadata store** consistency checking
- **Cache systems** (Redis, Dragonfly) coherence testing
- **Search indexes** synchronization validation

### Data Processing Pipelines
- **Document ingestion** quality validation
- **Embedding generation** consistency testing
- **Metadata extraction** accuracy verification
- **Content classification** validation
- **Search result relevance** quality testing

### External Dependencies
- **API response validation** for external services
- **Data format compatibility** testing
- **Service contract validation** for data exchanges
- **Error handling** for data quality failures

## Best Practices

### Test Design
- Use realistic test data that mirrors production scenarios
- Implement both positive and negative test cases
- Test edge cases and boundary conditions
- Validate both success and failure paths
- Include performance considerations in data quality tests

### Data Management
- Use isolated test databases to prevent interference
- Implement proper test data cleanup
- Create reusable test data factories
- Use deterministic test data for reproducible results
- Implement data versioning for test datasets

### Monitoring and Alerting
- Set up automated quality metric collection
- Implement threshold-based alerting
- Create quality trend dashboards
- Track quality regression over time
- Document quality issues and resolutions

## Tools and Frameworks

- **pytest**: Test framework with data quality fixtures
- **Great Expectations**: Data validation and profiling
- **pandas**: Data manipulation and validation
- **SQLAlchemy**: Database schema and constraint testing
- **Pydantic**: Data model validation and serialization

This data quality testing framework ensures reliable, consistent, and high-quality data throughout the AI Documentation Vector DB Hybrid Scraper system.