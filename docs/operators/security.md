# Security Guide

## Authentication & Authorization

### API Key Management
```bash
# Generate secure API keys
API_KEY=$(openssl rand -hex 32)
export AI_DOCS__API__SECRET_KEY="${API_KEY}"

# Create user with API key
USERNAME="production-user"
USER_API_KEY=$(openssl rand -hex 32)
redis-cli hset "user:${USERNAME}" \
  "api_key" "${USER_API_KEY}" \
  "role" "admin" \
  "created_at" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# List active API keys
redis-cli keys "user:*" | xargs -I {} redis-cli hgetall {}

# Rotate API keys
NEW_KEY=$(openssl rand -hex 32)
redis-cli hset "user:${USERNAME}" "api_key" "${NEW_KEY}"
```

### Rate Limiting
```bash
# Configure rate limits
export AI_DOCS__RATE_LIMIT__REQUESTS_PER_MINUTE=60
export AI_DOCS__RATE_LIMIT__BURST_SIZE=10

# Monitor rate limit violations
curl -H "X-API-Key: $API_KEY" \
     http://localhost:8000/health/rate-limits

# Check current quota
curl -H "X-API-Key: $API_KEY" \
     http://localhost:8000/health/quota
```

## Network Security

### Firewall Configuration
```bash
# Basic iptables rules
iptables -A INPUT -p tcp --dport 22 -j ACCEPT   # SSH
iptables -A INPUT -p tcp --dport 8000 -j ACCEPT # API
iptables -A INPUT -p tcp --dport 6333 -j DROP   # Qdrant (internal only)
iptables -A INPUT -p tcp --dport 6379 -j DROP   # Redis (internal only)
iptables -A INPUT -j DROP # Default deny

# UFW configuration (Ubuntu)
ufw allow 22/tcp
ufw allow 8000/tcp
ufw deny 6333/tcp
ufw deny 6379/tcp
ufw --force enable
```

### TLS/SSL Setup
```bash
# Generate SSL certificates
sudo certbot certonly --standalone -d your-domain.com

# Configure TLS
export AI_DOCS__TLS__ENABLED=true
export AI_DOCS__TLS__CERT_PATH="/etc/letsencrypt/live/your-domain.com/fullchain.pem"
export AI_DOCS__TLS__KEY_PATH="/etc/letsencrypt/live/your-domain.com/privkey.pem"

# Test TLS configuration
curl -I https://your-domain.com/health
```

### Reverse Proxy (Nginx)
```bash
# /etc/nginx/sites-available/ai-docs
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

# Enable configuration
ln -s /etc/nginx/sites-available/ai-docs /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx
```

## Container Security

### Secure Docker Configuration
```yaml
# docker-compose.yml security settings
services:
  api:
    user: "1000:1000"  # Non-root user
    read_only: true
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
    security_opt:
      - no-new-privileges:true
    tmpfs:
      - /tmp:noexec,nosuid,size=100m

  qdrant:
    user: "1000:1000"
    read_only: true
    cap_drop:
      - ALL
```

### Container Scanning
```bash
# Scan images for vulnerabilities
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image --severity HIGH,CRITICAL ai-docs:latest

# Scan running containers
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image --severity HIGH,CRITICAL $(docker ps --format "{{.Image}}")
```

## Data Protection

### Secrets Management
```bash
# Use Docker secrets instead of environment variables
echo "your-api-key" | docker secret create openai_api_key -
echo "your-db-password" | docker secret create db_password -

# Reference in docker-compose.yml
services:
  api:
    secrets:
      - openai_api_key
      - db_password
    environment:
      - OPENAI_API_KEY_FILE=/run/secrets/openai_api_key
      - DB_PASSWORD_FILE=/run/secrets/db_password

# Map secret files to the loader-prefixed variables during container startup, for example:
# export AI_DOCS__OPENAI__API_KEY="$(cat /run/secrets/openai_api_key)"
```

### Data Encryption
```bash
# Encrypt sensitive data at rest
export AI_DOCS__ENCRYPTION__ENABLED=true
export AI_DOCS__ENCRYPTION__KEY=$(openssl rand -hex 32)

# Database encryption
export QDRANT__STORAGE__ENCRYPTION__ENABLED=true
export REDIS__ENCRYPTION__ENABLED=true
```

### Backup Security
```bash
# Encrypt backups
BACKUP_DATE=$(date +%Y%m%d)
BACKUP_FILE="/backup/ai-docs-${BACKUP_DATE}.tar.gz"

# Create encrypted backup
tar -czf - /app/data | gpg --symmetric --cipher-algo AES256 > "${BACKUP_FILE}.gpg"

# Verify backup encryption
gpg --list-packets "${BACKUP_FILE}.gpg"
```

## Access Control

### User Roles
```bash
# Create role-based users
create_user() {
  local username=$1
  local role=$2
  local api_key=$(openssl rand -hex 32)
  
  redis-cli hset "user:${username}" \
    "api_key" "${api_key}" \
    "role" "${role}" \
    "created_at" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  
  echo "Created user: ${username} with role: ${role}"
  echo "API Key: ${api_key}"
}

# Create different user types
create_user "admin-user" "admin"
create_user "api-user" "api_access"
create_user "read-only-user" "viewer"
```

### Permission Management
```bash
# Check user permissions
check_permissions() {
  local username=$1
  local user_info=$(redis-cli hgetall "user:${username}")
  echo "User: ${username}"
  echo "Permissions: ${user_info}"
}

# Disable user account
disable_user() {
  local username=$1
  redis-cli hset "user:${username}" "status" "disabled"
  redis-cli hset "user:${username}" "disabled_at" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
```

## Security Monitoring

### Audit Logging
```bash
# Enable audit logging
export AI_DOCS__AUDIT__ENABLED=true
export AI_DOCS__AUDIT__LOG_PATH="/var/log/ai-docs/audit.log"

# Monitor suspicious activity
tail -f /var/log/ai-docs/audit.log | grep -E "(FAILED_AUTH|RATE_LIMIT|SUSPICIOUS)"

# Analyze failed authentication attempts
grep "FAILED_AUTH" /var/log/ai-docs/audit.log | \
  awk '{print $1, $2, $7}' | sort | uniq -c | sort -nr
```

### Security Alerts
```bash
# Monitor for security events
#!/bin/bash
# security-monitor.sh

ALERT_WEBHOOK="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
LOG_FILE="/var/log/ai-docs/audit.log"

# Check for suspicious activity
FAILED_AUTHS=$(grep -c "FAILED_AUTH" "$LOG_FILE" | tail -1)
if [ "$FAILED_AUTHS" -gt 10 ]; then
  curl -X POST "$ALERT_WEBHOOK" \
    -H 'Content-type: application/json' \
    --data "{\"text\":\"SECURITY ALERT: $FAILED_AUTHS failed authentication attempts\"}"
fi

# Check for rate limit violations
RATE_LIMITS=$(grep -c "RATE_LIMIT" "$LOG_FILE" | tail -1)
if [ "$RATE_LIMITS" -gt 50 ]; then
  curl -X POST "$ALERT_WEBHOOK" \
    -H 'Content-type: application/json' \
    --data "{\"text\":\"SECURITY ALERT: $RATE_LIMITS rate limit violations\"}"
fi
```

## Incident Response

### Security Incident Checklist
```bash
# 1. Immediate containment
docker-compose down  # Stop all services

# 2. Preserve evidence
cp /var/log/ai-docs/audit.log /backup/incident-$(date +%Y%m%d)/
docker logs api > /backup/incident-$(date +%Y%m%d)/api.log

# 3. Rotate all API keys
for user in $(redis-cli keys "user:*" | sed 's/user://'); do
  new_key=$(openssl rand -hex 32)
  redis-cli hset "user:${user}" "api_key" "${new_key}"
  echo "Rotated key for: ${user}"
done

# 4. Update firewall rules
iptables -P INPUT DROP
iptables -A INPUT -s TRUSTED_IP -j ACCEPT

# 5. Restart with new configuration
docker-compose up -d
```

### Recovery Procedures
```bash
# Secure recovery process
recovery_mode() {
  # Enable maintenance mode
  export AI_DOCS__MAINTENANCE_MODE=true
  
  # Restore from clean backup
  RECOVERY_DATE="20250322"  # Last known good
  ./scripts/secure-restore.sh "$RECOVERY_DATE"
  
  # Verify integrity
  ./scripts/security-audit.sh
  
  # Resume normal operations
  export AI_DOCS__MAINTENANCE_MODE=false
}
```

## Compliance

### Security Audit
```bash
# Run security audit
#!/bin/bash
# security-audit.sh

echo "Security Audit Report - $(date)"
echo "================================"

# Check for default passwords
echo "Checking for default credentials..."
if redis-cli auth password 2>/dev/null; then
  echo "WARNING: Default Redis password detected"
fi

# Check TLS configuration
echo "Checking TLS configuration..."
if ! curl -I https://localhost:8000/health 2>/dev/null; then
  echo "WARNING: TLS not properly configured"
fi

# Check file permissions
echo "Checking file permissions..."
find /app -type f -perm /o+w -exec echo "WARNING: World-writable file: {}" \;

# Check for exposed services
echo "Checking for exposed services..."
netstat -tlnp | grep -E ":6333|:6379" | grep "0.0.0.0" && \
  echo "WARNING: Database services exposed to external network"

echo "Security audit completed"
```
