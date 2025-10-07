# Security Documentation

Our security approach focuses on protecting user data, maintaining system integrity, and ensuring compliance with industry standards. We implement layered security measures including authentication, authorization, encryption, and continuous monitoring.

## Essential Security Documents

- [Security Essentials](./security-essentials.md) - Core security principles and fundamental practices
- [Security Checklist](./security-checklist.md) - Comprehensive checklist for security implementation

## Emergency Contact

**Security Incident Hotline:** [PHONE NUMBER]  
**Email:** [SECURITY EMAIL]  
**Hours:** 24/7

For immediate security threats or breaches, contact our security team directly using the information above.

## Agentic Runtime Controls

- **Network policy**: Browser automation pods run in isolated namespaces. Restrict outbound domains via `config/security.yml` (`allowed_domains`) and enforce with container firewall rules.
- **Credential handling**: Persistent secrets load from the environment or secret stores; automation flows read tokens via `AutomationCredentials` providers. Never embed credentials in tool definitions.
- **LLM guardrails**: `ToolExecutionService` enforces argument validation; upstream callers must validate payloads before routing to tools. Enable safety classifiers where supported by the selected LLM provider.
- **Adaptive rate limiting**: Middleware in FastAPI and FastMCP uses `purgatory-circuitbreaker` to trip on repeated failures. Thresholds are configured via `config/rate_limits.yml`.
- **Audit events**: LangGraph nodes emit structured telemetry with correlation IDs. Forward to the central log store and retain for 90 days for incident response.
- **Parallel execution limits**: Bound with `agentic.max_parallel_tools` to prevent resource exhaustion during coordinated attacks.

These controls replace the legacy planning guidance that previously lived under `planning/done/`.
