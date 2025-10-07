# Security Checklist

Use this checklist before deploying or promoting a new release.

## Access & Authentication

- [ ] All admin endpoints protected by authentication.
- [ ] API keys stored in environment variables or secret manager.
- [ ] Default credentials removed from `.env` and sample configs.
- [ ] Role-based access enforced for sensitive operations.

## Network & Infrastructure

- [ ] Allowed outbound domains set in `config/security.yml`.
- [ ] Ingress restricted to expected ports (8000 API, 6333 Qdrant, etc.).
- [ ] TLS certificates validated for external services.
- [ ] Browser automation pods run in isolated network namespace.

## Data Protection

- [ ] Qdrant snapshots encrypted and stored with restricted access.
- [ ] Redis/Dragonfly protected with authentication (if exposed).
- [ ] Backups verified and retention policy documented.
- [ ] Logs scrubbed for secrets before shipping to central store.

## Application Hardening

- [ ] Middleware rate limits configured in `config/rate_limits.yml`.
- [ ] LangGraph `agentic.max_parallel_tools` tuned to prevent exhaustion.
- [ ] Tool inputs validated (via `ToolExecutionService`) before execution.
- [ ] FastAPI CORS, CSP, and security headers enabled.

## Monitoring & Incident Response

- [ ] Prometheus alerts configured for authentication failures and 5xx spikes.
- [ ] Audit events forwarded with correlation IDs to log store.
- [ ] Security contact details verified in `docs/security/index.md`.
- [ ] Runbooks updated for latest release.

Document any deviations and obtain approval from the security owner before go-live.
