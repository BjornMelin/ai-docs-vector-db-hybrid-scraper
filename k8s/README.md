# Kubernetes Deployment Guide

This directory contains Kubernetes manifests for deploying the AI Docs Vector DB Hybrid Scraper system.

## Architecture

The deployment consists of:

- **Namespace**: `ai-docs-system` - Isolated environment for all components
- **Qdrant StatefulSet**: Vector database with persistent storage
- **DragonflyDB Deployment**: Redis-compatible cache
- **Application Deployment**: FastAPI web service (2 replicas)
- **Worker Deployment**: Background worker deployment template (2 replicas with HPA)
- **ConfigMap & Secrets**: Configuration and sensitive data management

## Prerequisites

- Kubernetes cluster (1.19+)
- kubectl configured
- Kustomize (optional, for customization)
- Storage class configured (default: `gp2`)
- Ingress controller (for external access)

## Quick Deployment

### 1. Prepare Local Secrets (Required)

This project uses Kustomize to generate secrets from local files, which **must not** be committed to version control.

1.  Create a `secrets` directory inside the `k8s` directory:
    ```bash
    mkdir -p k8s/secrets
    ```

2.  Create a file for each required secret. The filename becomes the key in the Kubernetes Secret.
    ```bash
    # Create the files with your actual secret values
    echo -n "your-openai-api-key" > k8s/secrets/OPENAI_API_KEY
    echo -n "your-anthropic-api-key" > k8s/secrets/ANTHROPIC_API_KEY
    echo -n "your-db-password" > k8s/secrets/DB_PASSWORD
    echo -n "your-super-secret-jwt-token" > k8s/secrets/JWT_SECRET
    ```
    The `k8s/secrets/` directory is already listed in `.gitignore` to prevent accidental commits.

### 2. Apply the Full Stack with Kustomize

Run the Kustomize build so the generated ConfigMap (`ai-docs-config`) and Secret (`ai-docs-secrets`) are created before the deployments start. This command also applies the namespace, storage, and deployment manifests referenced by `kustomization.yaml`.

```bash
kubectl apply -k .
```

> **Note:** This command is required because Kustomize adds a unique hash suffix to the generated ConfigMap and Secret. Applying the full stack ensures the deployments reference the correct generated resource names before any component starts.

### 3. (Optional) Apply Components Individually

If you prefer to apply individual manifests (for example, during debugging), make sure the previous `kubectl apply -k .` command has already been run so the generated resources exist.

```bash
# Create namespace (already included in the Kustomize apply)
kubectl apply -f namespace.yaml

# Deploy Qdrant vector database
kubectl apply -f qdrant-statefulset.yaml

# Deploy DragonflyDB cache
kubectl apply -f dragonfly-deployment.yaml

# Deploy main application
kubectl apply -f app-deployment.yaml

# Deploy background workers
kubectl apply -f worker-deployment.yaml
```

## Using Kustomize (Recommended)

### Development Deployment

```bash
kubectl apply -k .
```

### Production Deployment

```bash
# Update kustomization.yaml with your registry URLs
kubectl apply -k . --dry-run=client -o yaml | kubectl apply -f -
```

## Accessing the Application

### Port Forward (Development)

```bash
kubectl port-forward -n ai-docs-system svc/ai-docs-app 8000:8000
```

Access at: http://localhost:8000

### Ingress (Production)

Update `app-deployment.yaml` ingress section with your domain:

```yaml
spec:
  rules:
  - host: ai-docs.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ai-docs-app
            port:
              number: 8000
```

## Monitoring and Health Checks

### Check Pod Status

```bash
kubectl get pods -n ai-docs-system
```

### View Logs

```bash
# Application logs
kubectl logs -n ai-docs-system -l app.kubernetes.io/name=ai-docs-app -f

# Worker logs
kubectl logs -n ai-docs-system -l app.kubernetes.io/name=ai-docs-worker -f

# Database logs
kubectl logs -n ai-docs-system -l app.kubernetes.io/name=qdrant -f
```

### Health Check Endpoints

- Application: `http://ai-docs-app:8000/api/v1/config/status`
- Qdrant: `http://qdrant:6333/health`
- DragonflyDB: Redis PING command

## Scaling

### Manual Scaling

```bash
# Scale application pods
kubectl scale deployment -n ai-docs-system ai-docs-app --replicas=5

# Scale worker pods
kubectl scale deployment -n ai-docs-system ai-docs-worker --replicas=3
```

### Auto-scaling

Workers have HPA configured for CPU/memory-based scaling:

```bash
kubectl get hpa -n ai-docs-system
```

## Storage Management

### Persistent Volumes

- Qdrant: Uses StatefulSet with 50Gi storage (configurable)
- DragonflyDB: Uses PVC with 20Gi storage (configurable)

### Backup Considerations

- Qdrant data: `/qdrant/storage` (automatically backed up via snapshots)
- DragonflyDB: Redis persistence enabled with hourly snapshots

## Configuration

All non-sensitive configuration is managed declaratively within the `kustomization.yaml` file under the `configMapGenerator` section. To change a configuration value (e.g., `AI_DOCS_LOG_LEVEL`), edit the `literals` in this file directly. This ensures a single source of truth for configuration across all components.

Required secrets are managed via the `secretGenerator` in `kustomization.yaml`. See the "Create Local Secrets (Required)" section for instructions on providing secret values locally.

## Troubleshooting

### Common Issues

1. **Pods stuck in Pending**: Check resource quotas and node capacity
2. **ImagePullBackOff**: Verify image URLs and registry access
3. **CrashLoopBackOff**: Check logs and resource limits
4. **Service connectivity**: Verify network policies and DNS resolution

### Debug Commands

```bash
# Describe resources
kubectl describe pod -n ai-docs-system <pod-name>

# Check events
kubectl get events -n ai-docs-system --sort-by='.lastTimestamp'

# Test connectivity
kubectl exec -n ai-docs-system -it <pod-name> -- curl http://qdrant:6333/health
```

## Security

### Network Policies

Consider implementing network policies to restrict inter-pod communication:

```bash
# Example: Allow only app pods to access Qdrant
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: qdrant-access
  namespace: ai-docs-system
spec:
  podSelector:
    matchLabels:
      app.kubernetes.io/name: qdrant
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app.kubernetes.io/component: api
    - podSelector:
        matchLabels:
          app.kubernetes.io/component: worker
```

### Security Context

All containers run as non-root users (UID 1000) with restricted capabilities.

## Performance Tuning

### Resource Requests/Limits

Adjust based on your workload:

- **Development**: Lower limits for cost efficiency
- **Production**: Higher limits for performance (see `patches/production-resources.yaml`)

### Qdrant Optimization

Key environment variables for performance:

- `QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS`: CPU-based
- `QDRANT__STORAGE__HNSW__M`: Memory vs. accuracy trade-off
- `QDRANT__STORAGE__HNSW__EF_CONSTRUCT`: Index construction speed

### DragonflyDB Optimization

- `DRAGONFLY_THREADS`: Match CPU cores
- `DRAGONFLY_MEMORY_LIMIT`: 70-80% of pod memory limit
- `--compression=zstd`: Reduce memory usage

## Cleanup

```bash
# Delete all resources
kubectl delete namespace ai-docs-system

# Or selectively
kubectl delete -k .
```
