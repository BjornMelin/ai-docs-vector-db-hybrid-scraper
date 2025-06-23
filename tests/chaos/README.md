# Chaos Engineering Testing Suite

This directory contains chaos engineering and resilience testing for system reliability validation.

## Directory Structure

- **resilience/**: System resilience and recovery testing
- **fault_injection/**: Fault injection scenarios and testing
- **failure_scenarios/**: Failure simulation and recovery validation
- **network_chaos/**: Network-level chaos experiments
- **resource_exhaustion/**: Resource exhaustion and limit testing
- **dependency_failure/**: External dependency failure simulation

## Running Chaos Tests

```bash
# Run all chaos engineering tests
uv run pytest tests/chaos/ -v

# Run specific chaos category
uv run pytest tests/chaos/fault_injection/ -v

# Run with chaos markers
uv run pytest -m chaos -v
```

## Test Categories

### Resilience Testing (resilience/)
- System recovery validation
- Graceful degradation testing
- Circuit breaker functionality
- Retry mechanism validation

### Fault Injection (fault_injection/)
- Service fault simulation
- Database connection failures
- Memory allocation failures
- Disk I/O failures

### Failure Scenarios (failure_scenarios/)
- Complete service failure simulation
- Partial system failure testing
- Cascade failure prevention
- Disaster recovery validation

### Network Chaos (network_chaos/)
- Network latency injection
- Packet loss simulation
- Network partition testing
- Bandwidth limitation testing

### Resource Exhaustion (resource_exhaustion/)
- Memory exhaustion testing
- CPU saturation testing
- Disk space exhaustion
- Connection pool exhaustion

### Dependency Failure (dependency_failure/)
- External API failure simulation
- Database unavailability testing
- Cache service failure testing
- Message queue failure simulation

## Tools and Frameworks

- **chaos-engineering**: Python chaos engineering toolkit
- **pytest-chaos**: Chaos testing pytest integration
- **litmus**: Kubernetes chaos engineering
- **toxiproxy**: Network chaos proxy