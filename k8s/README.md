# Kubernetes Deployment Guide

This directory contains Kubernetes manifests for deploying the AI Docs Vector DB Hybrid Scraper system.

## Architecture

The deployment consists of:

- **Namespace**: `ai-docs-system` - Isolated environment for all components
- **Qdrant StatefulSet**: Vector database with persistent storage
- **DragonflyDB Deployment**: Redis-compatible cache
- **Application Deployment**: FastAPI web service (2 replicas)
- **ConfigMap & Secrets**: Configuration and sensitive data management

## Prerequisites

- Kubernetes cluster (1.19+)
- kubectl configured
- Kustomize (optional, for customization)
- Storage class configured (default: `gp2`)
- Ingress controller (for external access)

## Quick Deployment

### 1. Prepare Local Secrets (Required)

This project uses Kustomize to generate secrets from local files, which **must not** be committed to version control. Run these commands from the repository root.

1.  Create a `secrets` directory inside the `k8s` directory:
    ```bash
    mkdir -p k8s/secrets
    ```

2.  Create a file for each required secret. The filename becomes the environment variable that will be exposed to the pods.
    ```bash
    # Create the file with your actual secret value
    echo -n "your-openai-api-key" > k8s/secrets/AI_DOCS_OPENAI__API_KEY
    ```
    The `k8s/secrets/` directory is already listed in `.gitignore` to prevent accidental commits.

    > **Tip:** Add optional keys such as `AI_DOCS_BROWSER__FIRECRAWL__API_KEY` as additional files in the same directory when you enable those integrations.

### 2. Apply the Full Stack with Kustomize

Run the Kustomize build so the generated ConfigMap (`ai-docs-config`) and Secret (`ai-docs-secrets`) are created before the deployments start. This command also applies the namespace, storage, and deployment manifests referenced by `kustomization.yaml`.

```bash
kubectl apply -k k8s
```

> **Note:** This command is required because Kustomize adds a unique hash suffix to the generated ConfigMap and Secret. Applying the full stack ensures the deployments reference the correct generated resource names before any component starts and keeps the `envFrom` references in the deployments aligned with the hashed resource names.

### 3. (Optional) Inspect Components Individually

If you need to debug a single manifest, render it with Kustomize so the hashed resource names remain consistent:

```bash
# Render just the application deployment for inspection
kustomize build k8s | yq 'select(.metadata.name == "ai-docs-app")'
```

> **Important:** Apply changes through Kustomize (`kubectl apply -k k8s` or `kustomize build k8s | kubectl apply -f -`). Applying the raw manifests with `kubectl apply -f …` will overwrite the hashed ConfigMap and Secret references, leaving the deployments pointing at non-existent resources.

## Using Kustomize (Recommended)

### Development Deployment

```bash
kubectl apply -k k8s
```

### Production Deployment

```bash
# Update k8s/kustomization.yaml with your registry URLs
kubectl apply -k k8s --dry-run=client -o yaml | kubectl apply -f -
```

## Accessing the Application

### Port Forward (Development)

```bash
kubectl port-forward -n ai-docs-system svc/ai-docs-app 8000:8000
```

Access at: http://localhost:8000

### Ingress (Production)

Update `k8s/app-deployment.yaml` ingress section with your domain:

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

# Database logs
kubectl logs -n ai-docs-system -l app.kubernetes.io/name=qdrant -f
```

### Health Check Endpoints

- Application liveness: `http://ai-docs-app:8000/`
- Application readiness: `http://ai-docs-app:8000/health`
- Qdrant: `http://qdrant:6333/readyz`
- DragonflyDB: Redis PING command

## Scaling

### Manual Scaling

```bash
# Scale application pods
kubectl scale deployment -n ai-docs-system ai-docs-app --replicas=5
```

### Auto-scaling

Add a HorizontalPodAutoscaler for the application when production load requires it:

```bash
kubectl get hpa -n ai-docs-system
```

## Storage Management

### Persistent Volumes

- Qdrant: Uses StatefulSet with 50Gi storage (configurable)
- DragonflyDB: Uses PVC with 20Gi storage (configurable)

### Backup Considerations

- Qdrant data: `/qdrant/storage` on its PVC; configure external snapshots separately.
- DragonflyDB: The cache has no automatic snapshot schedule. The PVC is available for manually triggered snapshots; add the documented `--snapshot_cron` flag only when persistence is required.

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
kubectl exec -n ai-docs-system -it <pod-name> -- curl http://qdrant:6333/readyz
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
```

### Security Context

All containers run as non-root users (UID 1000) with restricted capabilities.

## Performance Tuning

### Resource Requests/Limits

Adjust based on your workload:

- **Development**: Lower limits for cost efficiency
- **Production**: Higher limits for performance (see `k8s/patches/production-resources.yaml`)

### Qdrant Optimization

Key environment variables for performance:

- `QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS`: CPU-based
- `QDRANT__STORAGE__HNSW__M`: Memory vs. accuracy trade-off
- `QDRANT__STORAGE__HNSW__EF_CONSTRUCT`: Index construction speed

### DragonflyDB Optimization

Dragonfly sizes its worker threads automatically. The base deployment caps cache memory at 3 GB within the 4 GiB pod limit. Keep the explicit `--maxmemory` value below the pod limit when tuning production resources.

## Cleanup

```bash
# Delete all resources
kubectl delete namespace ai-docs-system

# Or selectively
kubectl delete -k k8s
```
