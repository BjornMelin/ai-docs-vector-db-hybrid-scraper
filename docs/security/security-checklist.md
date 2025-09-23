# Security Checklist

Essential security measures for production deployment.

## Critical Pre-Deployment

### Secrets Management

- [ ] Move all API keys out of configuration files
- [ ] Generate new `FERNET_KEY`: `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`
- [ ] Store secrets in environment variables or secrets manager
- [ ] Verify no hardcoded secrets: `grep -r "sk-" src/`

### Dependency Security

- [ ] Run security audit: `uv run pip-audit`
- [ ] Address all HIGH and CRITICAL vulnerabilities
- [ ] Update dependencies: `uv sync --upgrade`
- [ ] Enable GitHub Dependabot

### Production Configuration

- [ ] Set `DEBUG=false` in all environments
- [ ] Remove development middleware in production
- [ ] Configure generic error responses
- [ ] Upgrade to AES-256 encryption if using AES-128

## Runtime Security

### Rate Limiting

- [ ] Migrate from in-memory to Redis-backed rate limiting
- [ ] Deploy behind Cloudflare WAF (free tier)
- [ ] Implement multi-layer rate limiting:
  - Edge Layer: Cloudflare WAF
  - Application Layer: Per-endpoint limits
  - User Layer: Per-user/IP limits

### Transport Security

- [ ] Enforce HTTPS with HSTS: `Strict-Transport-Security: max-age=31536000`
- [ ] Implement Content Security Policy (CSP)
- [ ] Configure security headers: X-Frame-Options, X-Content-Type-Options
- [ ] Verify SSL configuration: `openssl s_client -connect domain:443`

### Monitoring & Logging

- [ ] Implement structured JSON logging
- [ ] Log security events: failed auth, rate limits, validation failures
- [ ] Set up security monitoring and alerts
- [ ] Configure 90-day security log retention

## AI-Specific Security

### Prompt Injection Prevention

- [ ] Implement input/output fencing in SecurityValidator
- [ ] Deploy meta-prompt detection (regex + NLP)
- [ ] Block patterns: "Ignore previous instructions", "Act as if"
- [ ] Frame user queries: `[START_USER_INPUT]${query}[END_USER_INPUT]`

### Output Handling

- [ ] Treat LLM output as untrusted input
- [ ] Sanitize HTML, JavaScript, and active content
- [ ] Deploy PII detection with Microsoft Presidio
- [ ] Scan inputs before LLM processing and vector storage

### Vector Database Security

- [ ] Secure Qdrant with API keys or JWT (not anonymous)
- [ ] Restrict network access to application servers only
- [ ] Verify encryption at rest and in transit
- [ ] Implement read-only vs write access separation

## Data Protection

### Privacy Controls

- [ ] Deploy PII detection and redaction pipeline
- [ ] Implement 30-day auto-purge for user query logs
- [ ] Store only necessary metadata with embeddings
- [ ] Create privacy policy (`PRIVACY.md`)

### Access Controls

- [ ] Implement least privilege access to vector collections
- [ ] Separate credentials for read vs write operations
- [ ] Monitor embedding generation patterns
- [ ] Restrict embedding model API access

## Deployment Validation

### Security Testing

- [ ] Execute OWASP Top 10 compliance tests: `uv run pytest tests/security/compliance/`
- [ ] Test prompt injection attempts
- [ ] Verify PII extraction protection
- [ ] Test rate limiting effectiveness across restarts

### Production Verification

- [ ] Validate security headers: `curl -I https://domain.com`
- [ ] Test rate limiting: multiple rapid requests
- [ ] Verify HTTPS redirect: `curl -I http://domain.com`
- [ ] Check encryption status: no secrets in logs
- [ ] Confirm monitoring active: `curl domain.com/health/security`

## Compliance & Documentation

### Required Documentation

- [ ] Create vulnerability disclosure policy (`SECURITY.md`)
- [ ] Document data retention policies
- [ ] Update emergency contact information
- [ ] Create incident response procedures

### Success Metrics

- [ ] Zero hardcoded secrets in production
- [ ] 100% security tests passing
- [ ] <100ms PII scanning latency impact
- [ ] 99.9% uptime with rate limiting active
- [ ] Zero PII in logs or responses

## Emergency Response

### Incident Detection

- [ ] Monitor failed authentication attempts
- [ ] Alert on unusual query patterns
- [ ] Track rate limit violations
- [ ] Watch for suspicious activity patterns

### Response Procedures

1. Activate incident response plan
2. Isolate affected systems: `docker stop $(docker ps -q)`
3. Preserve logs: `cp /var/log/app/*.log /backup/incident-$(date +%Y%m%d)/`
4. Analyze: `grep -i "error\|fail\|attack" /var/log/app/security.log`
5. Document and review post-incident

### Emergency Contacts

- Security Lead: [CONFIGURE]
- Infrastructure: [CONFIGURE]
- Legal/Compliance: [CONFIGURE]
- 24/7 Hotline: [CONFIGURE]

---

**Review Schedule**: Monthly security checklist review  
**Last Updated**: 2025-01-10
