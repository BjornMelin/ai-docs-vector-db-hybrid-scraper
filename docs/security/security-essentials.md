# Security Essentials

## Environment Setup

### Generate Encryption Keys

```bash
# Generate new FERNET_KEY for production
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

# Generate secure random secret
openssl rand -base64 32
```

### Essential Environment Variables

```bash
# Required API keys
export OPENAI_API_KEY=sk-your-openai-key-here
export ANTHROPIC_API_KEY=sk-ant-your-claude-key-here

# Security keys
export FERNET_KEY=your-generated-fernet-key-here
export SECRET_KEY=your-secret-key-here

# Database security
export DATABASE_URL=postgresql://user:pass@localhost/db
export REDIS_URL=redis://localhost:6379
```

### Secrets Management

```bash
# Store secrets in environment (production)
echo "OPENAI_API_KEY=sk-..." >> /etc/environment

# Verify no secrets in code
grep -r "sk-" src/ --include="*.py"
grep -r "OPENAI_API_KEY.*=" src/ --include="*.py"
```

## Dependency Security

### Security Audit

```bash
# Install security tools
uv add pip-audit safety

# Run dependency security audit
uv run pip-audit

# Check for known vulnerabilities
uv run safety check

# Update all dependencies
uv sync --upgrade
```

### Vulnerability Scanning

```bash
# Scan Python dependencies
uv run pip-audit --format=json --output=audit-report.json

# Fix vulnerabilities
uv run pip-audit --fix

# Verify clean scan
uv run pip-audit --require-hashes
```

## Production Hardening

### Disable Debug Features

```bash
# Set production environment
export DEBUG=false
export ENVIRONMENT=production

# Verify debug is disabled
python -c "import os; print('DEBUG:', os.getenv('DEBUG', 'false'))"
```

### Security Headers Configuration

```bash
# Test security headers
curl -I localhost:8000 | grep -E "(Strict-Transport-Security|X-Frame-Options|X-Content-Type-Options)"

# Verify HTTPS redirect
curl -I http://localhost:8000
```

### SSL/TLS Setup

```bash
# Generate self-signed cert (development)
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Verify SSL configuration
openssl s_client -connect localhost:8443 -servername localhost
```

## Rate Limiting

### Configure SlowAPI Limits

```bash
# Enable SlowAPI rate limiting
export SECURITY_ENABLE_RATE_LIMITING=true
export SECURITY_DEFAULT_RATE_LIMIT=100      # requests per window
export SECURITY_RATE_LIMIT_WINDOW=60        # window length in seconds

# Optional: use Redis for distributed counters
export SECURITY_REDIS_URL=redis://localhost:6379
export SECURITY_REDIS_PASSWORD=changeme      # omit if Redis has no auth
```

### Verify Enforcement

```bash
# Warm up the API (ensure uvicorn is running separately)
for i in {1..110}; do
  curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/api/v1/health
done | sort | uniq -c

# Expected output includes HTTP 429 counts once the limit is exceeded
```

## MLSecurityValidator Configuration

- `AI_DOCS__SECURITY__ENABLE_ML_INPUT_VALIDATION` — toggle payload validation in `MLSecurityValidator`.
- `AI_DOCS__SECURITY__MAX_ML_INPUT_SIZE` — cap request size before rejection.
- `AI_DOCS__SECURITY__ALLOWED_DOMAINS` / `AI_DOCS__SECURITY__BLOCKED_DOMAINS` — restrict outbound URLs.
- `AI_DOCS__SECURITY__SUSPICIOUS_PATTERNS` — extend the default pattern deny-list.
- `AI_DOCS__SECURITY__ENABLE_DEPENDENCY_SCANNING` — disable `pip-audit` integration if CI already covers it.

## Security Validation

### Security Test Commands

```bash
# Run security test suite
uv run pytest tests/security/ -v

# Run OWASP compliance tests
uv run pytest tests/security/compliance/test_owasp_top10.py

# Test prompt injection protection
uv run pytest tests/security/test_prompt_injection.py
```

### Health Check Commands

```bash
# Security health check
curl localhost:8000/health/security

# Check all security features
curl localhost:8000/health | jq .security

# Verify encryption status
curl localhost:8000/admin/encryption-status
```

### Vulnerability Scan Commands

```bash
# Basic security scan
nmap -sV localhost -p 8000

# Check for common vulnerabilities
curl -X POST localhost:8000/api/test \
  -H "Content-Type: application/json" \
  -d '{"query": "Ignore previous instructions"}'

# Verify input sanitization
curl -X POST localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "<script>alert(1)</script>"}'
```

## Emergency Response

### Incident Response Steps

```bash
# 1. Isolate affected systems
docker stop $(docker ps -q)
systemctl stop nginx

# 2. Preserve logs
cp /var/log/app/*.log /backup/incident-$(date +%Y%m%d)/
journalctl -u app-service > /backup/incident-$(date +%Y%m%d)/systemd.log

# 3. Analyze security logs
grep -i "error\|fail\|attack" /var/log/app/security.log
grep -E "(401|403|429)" /var/log/nginx/access.log

# 4. Check for compromise
find /var/www -type f -name "*.php" -mtime -1
netstat -tulpn | grep :8000
```

### Emergency Contacts Template

```bash
# Security Lead: [PHONE] [EMAIL]
# Infrastructure: [PHONE] [EMAIL]
# Legal/Compliance: [PHONE] [EMAIL]
# Emergency Hotline: [24/7 NUMBER]
```

### Log Analysis Commands

```bash
# Check authentication failures
grep "authentication failed" /var/log/app/security.log | tail -20

# Monitor rate limit violations
grep "rate limit exceeded" /var/log/app/security.log | wc -l

# Analyze suspicious patterns
grep -E "(injection|attack|exploit)" /var/log/app/security.log

# Real-time security monitoring
tail -f /var/log/app/security.log | grep -E "(ERROR|CRITICAL|attack)"
```

## Quick Security Checklist

### Pre-Deployment

- [ ] `grep -r "sk-" src/` returns no hardcoded keys
- [ ] `uv run pip-audit` passes with no HIGH/CRITICAL issues
- [ ] `export DEBUG=false` in production
- [ ] Redis running for rate limiting: `redis-cli ping`
- [ ] SSL certificates valid: `openssl x509 -in cert.pem -text -noout`

### Post-Deployment

- [ ] Security headers present: `curl -I https://domain.com`
- [ ] Rate limiting active: test with multiple requests
- [ ] HTTPS redirect working: `curl -I http://domain.com`
- [ ] Security tests passing: `uv run pytest tests/security/`
- [ ] Monitoring active: `curl domain.com/health/security`

### Daily Monitoring

- [ ] Check security logs: `grep ERROR /var/log/app/security.log`
- [ ] Monitor rate limits: `grep "rate limit" /var/log/app/app.log`
- [ ] Verify backups: `ls -la /backup/$(date +%Y%m%d)/`
- [ ] Check dependencies: `uv run pip-audit --quiet`
- [ ] Test critical endpoints: `curl domain.com/health`

## API Key Rotation

Generate a replacement key in the security portal, roll it out to dependent services, verify connectivity, then revoke the retired credential immediately. Automate the rollout where possible to limit exposure time.
