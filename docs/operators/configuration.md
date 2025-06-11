# Configuration Management Guide

> **Status**: Current  
> **Last Updated**: 2025-06-09  
> **Purpose**: Comprehensive configuration guide for operators  
> **Audience**: DevOps engineers, system administrators, and operators

This guide provides comprehensive configuration management for operators deploying and maintaining the AI Documentation Vector DB system across different environments.

## Table of Contents

- [Overview](#overview)
- [Configuration Architecture](#configuration-architecture)
- [Environment Management](#environment-management)
- [Production Configuration](#production-configuration)
- [Development Configuration](#development-configuration)
- [Security Configuration](#security-configuration)
- [Performance Tuning](#performance-tuning)
- [Configuration Validation](#configuration-validation)
- [Change Management](#change-management)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)
- [Monitoring and Alerting](#monitoring-and-alerting)

## Overview

The AI Documentation Vector DB uses a unified configuration system built with Pydantic v2 that provides:

- **Single Source of Truth**: All configuration in one `UnifiedConfig` class
- **Type Safety**: Full validation and type checking
- **Multiple Sources**: Support for files, environment variables, and code
- **Environment-Specific**: Templates for different deployment environments
- **Migration Tools**: Automatic migration between configuration versions
- **CLI Management**: Rich CLI for configuration operations

### Key Benefits for Operators

- **Standardized Deployment**: Consistent configuration across environments
- **Security**: Built-in validation and security checks
- **Monitoring**: Comprehensive configuration health monitoring
- **Automation**: CLI tools for configuration management tasks
- **Scalability**: Performance tuning based on environment requirements

## Configuration Architecture

```mermaid
graph TB
    A[Configuration Sources] --> B[UnifiedConfig]
    
    A1[Environment Variables] --> A
    A2[Configuration Files] --> A
    A3[.env Files] --> A
    A4[Default Values] --> A
    
    B --> C[Component Configurations]
    
    C --> C1[Cache Config]
    C --> C2[Qdrant Config]
    C --> C3[Provider Configs]
    C --> C4[Performance Config]
    C --> C5[Security Config]
    
    B --> D[Configuration Management]
    D --> D1[Validation]
    D --> D2[Migration]
    D --> D3[Templates]
    D --> D4[CLI Tools]
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#fff3e0
    style D fill:#e8f5e8
```

### Configuration Hierarchy

1. **Environment Variables** (highest priority)
2. **Configuration Files** (JSON, YAML, TOML)
3. **`.env` Files**
4. **Default Values** (lowest priority)

## Environment Management

### Environment Templates

The system provides pre-configured templates for different environments:

| Template | Use Case | Key Features |
|----------|----------|--------------|
| `production.json` | Production deployment | High performance, security, monitoring |
| `development.json` | Local development | Debug features, local services |
| `local-only.json` | Privacy-focused | No cloud services, local-only |
| `testing.json` | Test execution | Optimized for test speed |
| `minimal.json` | Basic deployment | Minimal configuration with defaults |

### Environment-Specific Configuration

#### Production Environment

```bash
# Set environment
export AI_DOCS__ENVIRONMENT=production
export CONFIG_FILE=config/production.json

# Required environment variables
export AI_DOCS__OPENAI__API_KEY=${OPENAI_API_KEY}
export AI_DOCS__FIRECRAWL__API_KEY=${FIRECRAWL_API_KEY}
export AI_DOCS__CACHE__DRAGONFLY_URL="redis://dragonfly:6379"
export AI_DOCS__QDRANT__URL="http://qdrant:6333"
```

#### Staging Environment

```bash
# Set environment
export AI_DOCS__ENVIRONMENT=staging
export CONFIG_FILE=config/staging.json

# Staging-specific overrides
export AI_DOCS__DEBUG=false
export AI_DOCS__LOG_LEVEL=INFO
export AI_DOCS__PERFORMANCE__MAX_CONCURRENT_REQUESTS=10
```

#### Development Environment

```bash
# Set environment
export AI_DOCS__ENVIRONMENT=development
export CONFIG_FILE=config/development.json

# Development overrides
export AI_DOCS__DEBUG=true
export AI_DOCS__LOG_LEVEL=DEBUG
export AI_DOCS__EMBEDDING_PROVIDER=fastembed
export AI_DOCS__CACHE__ENABLE_DRAGONFLY_CACHE=false
```

## Production Configuration

### Complete Production Template

```json
{
  "environment": "production",
  "debug": false,
  "log_level": "INFO",
  "embedding_provider": "openai",
  "crawl_provider": "crawl4ai",
  
  "cache": {
    "enable_caching": true,
    "enable_local_cache": true,
    "enable_dragonfly_cache": true,
    "dragonfly_url": "redis://dragonfly:6379",
    "ttl_embeddings": 86400,
    "ttl_crawl": 3600,
    "ttl_queries": 7200,
    "local_max_size": 1000,
    "local_max_memory_mb": 100.0,
    "redis_pool_size": 20
  },
  
  "qdrant": {
    "url": "http://qdrant:6333",
    "collection_name": "documents",
    "batch_size": 100,
    "max_retries": 5,
    "hnsw_ef_construct": 200,
    "hnsw_m": 16,
    "quantization_enabled": true
  },
  
  "openai": {
    "model": "text-embedding-3-small",
    "dimensions": 1536,
    "batch_size": 100,
    "max_requests_per_minute": 3000,
    "cost_per_million_tokens": 0.02,
    "budget_limit": 100.0
  },
  
  "crawl4ai": {
    "enable_memory_adaptive_dispatcher": true,
    "memory_threshold_percent": 75.0,
    "max_session_permit": 25,
    "dispatcher_check_interval": 1.0,
    "headless": true,
    "max_concurrent_crawls": 10,
    "page_timeout": 30.0,
    "viewport": {"width": 1920, "height": 1080},
    "remove_scripts": true,
    "remove_styles": true,
    "enable_streaming": true,
    "rate_limit_base_delay_min": 1.0,
    "rate_limit_base_delay_max": 2.0,
    "rate_limit_max_delay": 30.0,
    "rate_limit_max_retries": 2
  },
  
  "chunking": {
    "strategy": "enhanced",
    "chunk_size": 1600,
    "chunk_overlap": 200,
    "preserve_function_boundaries": true
  },
  
  "performance": {
    "max_concurrent_requests": 20,
    "request_timeout": 30.0,
    "max_retries": 3,
    "retry_base_delay": 1.0,
    "retry_max_delay": 60.0,
    "max_memory_usage_mb": 2000.0,
    "gc_threshold": 0.8,
    "default_rate_limits": {
      "openai": {"max_calls": 500, "time_window": 60},
      "firecrawl": {"max_calls": 100, "time_window": 60},
      "crawl4ai": {"max_calls": 50, "time_window": 1},
      "qdrant": {"max_calls": 100, "time_window": 1}
    }
  },
  
  "security": {
    "require_api_keys": true,
    "enable_rate_limiting": true,
    "rate_limit_requests": 100,
    "allowed_domains": ["docs.python.org", "fastapi.tiangolo.com"],
    "blocked_domains": ["malicious.example.com"]
  }
}
```

### High-Performance Production Configuration

For systems with 16GB+ RAM and high throughput requirements:

```json
{
  "crawl4ai": {
    "enable_memory_adaptive_dispatcher": true,
    "memory_threshold_percent": 80.0,
    "max_session_permit": 50,
    "dispatcher_check_interval": 0.5,
    "enable_streaming": true,
    "rate_limit_base_delay_min": 0.1,
    "rate_limit_base_delay_max": 0.5,
    "rate_limit_max_delay": 10.0,
    "rate_limit_max_retries": 3
  },
  
  "performance": {
    "max_concurrent_requests": 50,
    "max_memory_usage_mb": 4000.0,
    "default_rate_limits": {
      "openai": {"max_calls": 1000, "time_window": 60},
      "crawl4ai": {"max_calls": 100, "time_window": 1}
    }
  },
  
  "cache": {
    "redis_pool_size": 50,
    "local_max_size": 2000,
    "local_max_memory_mb": 200.0
  }
}
```

### Memory-Constrained Production Configuration

For systems with 8GB or less RAM:

```json
{
  "crawl4ai": {
    "enable_memory_adaptive_dispatcher": true,
    "memory_threshold_percent": 60.0,
    "max_session_permit": 5,
    "dispatcher_check_interval": 2.0,
    "enable_streaming": false
  },
  
  "performance": {
    "max_concurrent_requests": 5,
    "max_memory_usage_mb": 1000.0,
    "default_rate_limits": {
      "crawl4ai": {"max_calls": 10, "time_window": 1}
    }
  },
  
  "cache": {
    "local_max_size": 100,
    "local_max_memory_mb": 25.0,
    "redis_pool_size": 5
  }
}
```

## Development Configuration

### Local Development Setup

```json
{
  "environment": "development",
  "debug": true,
  "log_level": "DEBUG",
  "embedding_provider": "fastembed",
  "crawl_provider": "crawl4ai",
  
  "cache": {
    "enable_caching": true,
    "enable_local_cache": true,
    "enable_dragonfly_cache": false,
    "local_max_size": 500,
    "local_max_memory_mb": 50.0
  },
  
  "qdrant": {
    "url": "http://localhost:6333",
    "collection_name": "dev_documents",
    "batch_size": 50
  },
  
  "fastembed": {
    "model": "BAAI/bge-small-en-v1.5",
    "batch_size": 16
  },
  
  "crawl4ai": {
    "headless": false,
    "max_concurrent_crawls": 5,
    "page_timeout": 60.0,
    "enable_memory_adaptive_dispatcher": true,
    "memory_threshold_percent": 70.0,
    "max_session_permit": 10
  },
  
  "performance": {
    "max_concurrent_requests": 5,
    "request_timeout": 60.0,
    "max_memory_usage_mb": 500.0
  },
  
  "security": {
    "require_api_keys": false,
    "enable_rate_limiting": false
  }
}
```

## Security Configuration

### API Key Management

```bash
# Production secrets (never commit to version control)
export AI_DOCS__OPENAI__API_KEY="sk-your-production-key"
export AI_DOCS__FIRECRAWL__API_KEY="fc-your-production-key"

# Use different keys for different environments
export AI_DOCS__OPENAI__API_KEY="${OPENAI_API_KEY_PROD}"
export AI_DOCS__FIRECRAWL__API_KEY="${FIRECRAWL_API_KEY_PROD}"
```

### Security Hardening Configuration

```json
{
  "security": {
    "require_api_keys": true,
    "enable_rate_limiting": true,
    "rate_limit_requests": 100,
    "rate_limit_window": 60,
    "allowed_domains": [
      "docs.python.org",
      "fastapi.tiangolo.com",
      "docs.pydantic.dev"
    ],
    "blocked_domains": [
      "malicious.com",
      "spam.example.org"
    ],
    "max_content_length": 10485760,
    "enable_cors": true,
    "cors_origins": ["https://yourdomain.com"],
    "enable_ssl": true,
    "ssl_verify": true
  }
}
```

### Environment-Specific Security

#### Production Security

```json
{
  "security": {
    "require_api_keys": true,
    "enable_rate_limiting": true,
    "rate_limit_requests": 100,
    "enable_ssl": true,
    "ssl_verify": true,
    "enable_cors": false,
    "log_security_events": true
  }
}
```

#### Development Security

```json
{
  "security": {
    "require_api_keys": false,
    "enable_rate_limiting": false,
    "enable_ssl": false,
    "ssl_verify": false,
    "enable_cors": true,
    "cors_origins": ["http://localhost:3000"]
  }
}
```

## Enhanced Database Connection Pool Configuration (BJO-134)

### Database Connection Pool Settings

#### 1. Basic Connection Pool Configuration

```json
{
  "database": {
    "host": "localhost",
    "port": 5432,
    "name": "ai_docs_db",
    "connection_pool": {
      "min_size": 5,
      "max_size": 20,
      "max_overflow": 10,
      "pool_timeout": 30,
      "pool_recycle": 3600,
      "pool_pre_ping": true
    }
  }
}
```

#### 2. Enhanced Features Configuration

```json
{
  "database": {
    "enhanced_features": {
      "enable_predictive_monitoring": true,
      "enable_connection_affinity": true,
      "enable_adaptive_config": true,
      "enable_ml_optimization": true
    },
    "predictive_monitoring": {
      "model_type": "random_forest",
      "training_interval_hours": 24,
      "prediction_horizon_minutes": 15,
      "feature_window_minutes": 60,
      "min_training_samples": 1000,
      "model_accuracy_threshold": 0.7,
      "retrain_on_accuracy_drop": true
    },
    "circuit_breaker": {
      "failure_threshold": 5,
      "recovery_timeout_seconds": 60,
      "half_open_max_calls": 3,
      "failure_types": {
        "connection": {"threshold": 3, "timeout": 30},
        "query": {"threshold": 5, "timeout": 60},
        "transaction": {"threshold": 2, "timeout": 120}
      }
    },
    "connection_affinity": {
      "max_patterns": 1000,
      "max_connections": 50,
      "pattern_ttl_hours": 24,
      "min_pattern_executions": 5,
      "affinity_score_threshold": 0.3,
      "enable_query_normalization": true
    },
    "adaptive_config": {
      "strategy": "moderate",
      "adaptation_interval_minutes": 15,
      "enable_auto_scaling": true,
      "scaling_factors": {
        "cpu_threshold": 0.8,
        "memory_threshold": 0.85,
        "connection_threshold": 0.9
      }
    }
  }
}
```

### Environment-Specific Database Configurations

#### 1. Development Environment

```json
{
  "database": {
    "connection_pool": {
      "min_size": 2,
      "max_size": 5,
      "max_overflow": 2
    },
    "enhanced_features": {
      "enable_predictive_monitoring": false,
      "enable_connection_affinity": true,
      "enable_adaptive_config": false,
      "enable_ml_optimization": false
    },
    "predictive_monitoring": {
      "model_type": "simple_linear",
      "training_interval_hours": 168
    }
  }
}
```

#### 2. Production Environment

```json
{
  "database": {
    "connection_pool": {
      "min_size": 10,
      "max_size": 50,
      "max_overflow": 20,
      "pool_timeout": 30,
      "pool_recycle": 1800
    },
    "enhanced_features": {
      "enable_predictive_monitoring": true,
      "enable_connection_affinity": true,
      "enable_adaptive_config": true,
      "enable_ml_optimization": true
    },
    "predictive_monitoring": {
      "model_type": "random_forest",
      "training_interval_hours": 6,
      "prediction_horizon_minutes": 30,
      "feature_window_minutes": 120,
      "min_training_samples": 5000,
      "model_accuracy_threshold": 0.8
    },
    "circuit_breaker": {
      "failure_threshold": 3,
      "recovery_timeout_seconds": 30,
      "half_open_max_calls": 5
    },
    "connection_affinity": {
      "max_patterns": 5000,
      "max_connections": 100,
      "pattern_ttl_hours": 168,
      "min_pattern_executions": 10
    }
  }
}
```

#### 3. High-Performance Environment

```json
{
  "database": {
    "connection_pool": {
      "min_size": 20,
      "max_size": 100,
      "max_overflow": 50,
      "pool_timeout": 15,
      "pool_recycle": 900
    },
    "enhanced_features": {
      "enable_predictive_monitoring": true,
      "enable_connection_affinity": true,
      "enable_adaptive_config": true,
      "enable_ml_optimization": true
    },
    "predictive_monitoring": {
      "model_type": "gradient_boosting",
      "training_interval_hours": 2,
      "prediction_horizon_minutes": 45,
      "feature_window_minutes": 180,
      "min_training_samples": 10000,
      "model_accuracy_threshold": 0.85
    },
    "adaptive_config": {
      "strategy": "aggressive",
      "adaptation_interval_minutes": 5,
      "enable_auto_scaling": true
    }
  }
}
```

### Configuration Validation and Testing

#### 1. Database Configuration Validation

```bash
#!/bin/bash
# Validate enhanced database configuration
echo "=== Enhanced Database Configuration Validation ==="

# 1. Validate basic configuration structure
uv run python -c "
from src.config.loader import UnifiedConfig
try:
    config = UnifiedConfig.load_from_file('config.json')
    print('✓ Configuration structure valid')
    print(f'  Database host: {config.database.host}')
    print(f'  Pool size: {config.database.connection_pool.min_size}-{config.database.connection_pool.max_size}')
except Exception as e:
    print(f'✗ Configuration validation failed: {e}')
    exit(1)
"

# 2. Test database connectivity
echo "Testing database connectivity..."
uv run python -c "
import asyncio
from src.infrastructure.database.connection_manager import AsyncConnectionManager
from src.config.loader import UnifiedConfig

async def test_connection():
    config = UnifiedConfig.load_from_file('config.json')
    manager = AsyncConnectionManager(config.database)
    try:
        await manager.initialize()
        print('✓ Database connection successful')
        await manager.shutdown()
    except Exception as e:
        print(f'✗ Database connection failed: {e}')
        return False
    return True

if not asyncio.run(test_connection()):
    exit(1)
"

# 3. Validate enhanced features
echo "Validating enhanced features..."
uv run python scripts/validate_enhanced_db_config.py --config config.json

echo "✓ Enhanced database configuration validation completed"
```

#### 2. Performance Configuration Testing

```python
#!/usr/bin/env python3
"""Test enhanced database configuration performance."""

import asyncio
import time
from typing import Dict, Any

from src.config.loader import UnifiedConfig
from src.infrastructure.database.connection_manager import AsyncConnectionManager

class DatabaseConfigTester:
    """Test database configuration performance."""
    
    def __init__(self, config_file: str):
        self.config = UnifiedConfig.load_from_file(config_file)
        
    async def test_performance_targets(self) -> Dict[str, Any]:
        """Test that configuration meets performance targets."""
        manager = AsyncConnectionManager(self.config.database)
        
        try:
            await manager.initialize()
            
            # Test connection pool performance
            start_time = time.time()
            tasks = []
            for _ in range(20):
                task = manager.execute_query("SELECT 1")
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            end_time = time.time()
            
            avg_latency = (end_time - start_time) / len(tasks)
            
            # Performance targets
            targets = {
                'avg_latency_ms': avg_latency * 1000,
                'target_latency_ms': 100,  # 100ms target
                'meets_target': avg_latency < 0.1,
                'pool_utilization': await self._get_pool_utilization(manager),
                'enhanced_features_active': self._check_enhanced_features()
            }
            
            return targets
            
        finally:
            await manager.shutdown()
    
    def _check_enhanced_features(self) -> Dict[str, bool]:
        """Check which enhanced features are enabled."""
        features = self.config.database.enhanced_features
        return {
            'predictive_monitoring': features.enable_predictive_monitoring,
            'connection_affinity': features.enable_connection_affinity,
            'adaptive_config': features.enable_adaptive_config,
            'ml_optimization': features.enable_ml_optimization
        }

if __name__ == "__main__":
    import sys
    
    tester = DatabaseConfigTester(sys.argv[1] if len(sys.argv) > 1 else "config.json")
    results = asyncio.run(tester.test_performance_targets())
    
    print("Performance Test Results:")
    print(f"  Average Latency: {results['avg_latency_ms']:.2f}ms")
    print(f"  Target Latency: {results['target_latency_ms']}ms")
    print(f"  Meets Target: {'✓' if results['meets_target'] else '✗'}")
    print(f"  Enhanced Features: {results['enhanced_features_active']}")
    
    if not results['meets_target']:
        sys.exit(1)
```

### Configuration Migration and Upgrades

#### 1. Migration to Enhanced Database Features

```bash
#!/bin/bash
# Migrate existing configuration to enhanced database features
echo "=== Enhanced Database Configuration Migration ==="

# 1. Backup existing configuration
cp config.json config.json.backup.$(date +%Y%m%d_%H%M%S)

# 2. Create enhanced configuration
uv run python -c "
from src.config.loader import UnifiedConfig
from src.config.migration import ConfigMigrator

# Load existing config
config = UnifiedConfig.load_from_file('config.json')

# Apply enhanced database migration
migrator = ConfigMigrator()
enhanced_config = migrator.add_enhanced_database_features(config)

# Save enhanced configuration
enhanced_config.save_to_file('config.enhanced.json')
print('✓ Enhanced configuration created: config.enhanced.json')
"

# 3. Validate enhanced configuration
uv run python scripts/validate_enhanced_db_config.py --config config.enhanced.json

# 4. Test performance with enhanced features
echo "Testing enhanced configuration performance..."
uv run python scripts/test_enhanced_db_performance.py --config config.enhanced.json

# 5. Apply enhanced configuration
if [ $? -eq 0 ]; then
    mv config.enhanced.json config.json
    echo "✓ Enhanced database configuration applied successfully"
else
    echo "✗ Enhanced configuration failed validation"
    exit 1
fi
```

#### 2. Environment Variable Override Support

```bash
# Enhanced database configuration environment variables

# Basic connection pool settings
export DB_CONNECTION_POOL_MIN_SIZE=10
export DB_CONNECTION_POOL_MAX_SIZE=50
export DB_CONNECTION_POOL_MAX_OVERFLOW=20

# Enhanced features toggles
export DB_ENABLE_PREDICTIVE_MONITORING=true
export DB_ENABLE_CONNECTION_AFFINITY=true
export DB_ENABLE_ADAPTIVE_CONFIG=true
export DB_ENABLE_ML_OPTIMIZATION=true

# ML model settings
export DB_ML_MODEL_TYPE=random_forest
export DB_ML_TRAINING_INTERVAL_HOURS=24
export DB_ML_PREDICTION_HORIZON_MINUTES=15
export DB_ML_MODEL_ACCURACY_THRESHOLD=0.7

# Circuit breaker settings
export DB_CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
export DB_CIRCUIT_BREAKER_RECOVERY_TIMEOUT=60
export DB_CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS=3

# Connection affinity settings
export DB_CONNECTION_AFFINITY_MAX_PATTERNS=1000
export DB_CONNECTION_AFFINITY_MAX_CONNECTIONS=50
export DB_CONNECTION_AFFINITY_PATTERN_TTL_HOURS=24

# Adaptive configuration settings
export DB_ADAPTIVE_CONFIG_STRATEGY=moderate
export DB_ADAPTIVE_CONFIG_INTERVAL_MINUTES=15
export DB_ADAPTIVE_CONFIG_ENABLE_AUTO_SCALING=true
```

### Configuration Monitoring and Health Checks

#### 1. Configuration Health Monitoring

```python
#!/usr/bin/env python3
"""Monitor enhanced database configuration health."""

import asyncio
from typing import Dict, List

from src.config.loader import UnifiedConfig
from src.infrastructure.database.connection_manager import AsyncConnectionManager

class ConfigHealthMonitor:
    """Monitor configuration health and performance."""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        
    async def check_configuration_health(self) -> Dict[str, Any]:
        """Comprehensive configuration health check."""
        health_report = {
            'overall_status': 'healthy',
            'checks': {},
            'recommendations': []
        }
        
        # 1. Database connection health
        health_report['checks']['database_connection'] = await self._check_database_connection()
        
        # 2. Enhanced features health
        health_report['checks']['enhanced_features'] = await self._check_enhanced_features()
        
        # 3. Performance metrics validation
        health_report['checks']['performance_metrics'] = await self._check_performance_metrics()
        
        # 4. Configuration consistency
        health_report['checks']['config_consistency'] = self._check_config_consistency()
        
        # Generate recommendations
        health_report['recommendations'] = self._generate_recommendations(health_report['checks'])
        
        # Determine overall status
        failed_checks = [name for name, result in health_report['checks'].items() 
                        if not result.get('healthy', False)]
        if failed_checks:
            health_report['overall_status'] = 'unhealthy'
            health_report['failed_checks'] = failed_checks
            
        return health_report
```

## Performance Tuning

### CPU-Optimized Configuration

```json
{
  "performance": {
    "max_concurrent_requests": 50,
    "request_timeout": 15.0,
    "max_retries": 2,
    "retry_base_delay": 0.5,
    "retry_max_delay": 30.0,
    "gc_threshold": 0.9,
    "default_rate_limits": {
      "openai": {"max_calls": 1000, "time_window": 60},
      "crawl4ai": {"max_calls": 100, "time_window": 1}
    }
  },
  
  "crawl4ai": {
    "max_concurrent_crawls": 20,
    "page_timeout": 15.0,
    "enable_memory_adaptive_dispatcher": true,
    "memory_threshold_percent": 85.0,
    "max_session_permit": 100
  }
}
```

### Memory-Optimized Configuration

```json
{
  "performance": {
    "max_concurrent_requests": 10,
    "max_memory_usage_mb": 1000.0,
    "gc_threshold": 0.7,
    "enable_memory_monitoring": true,
    "memory_check_interval": 30.0
  },
  
  "cache": {
    "local_max_size": 100,
    "local_max_memory_mb": 50.0,
    "enable_memory_pressure_eviction": true
  },
  
  "crawl4ai": {
    "enable_memory_adaptive_dispatcher": true,
    "memory_threshold_percent": 60.0,
    "max_session_permit": 5,
    "enable_streaming": false
  }
}
```

### Network-Optimized Configuration

```json
{
  "performance": {
    "request_timeout": 10.0,
    "max_retries": 5,
    "retry_base_delay": 0.1,
    "connection_pool_size": 100,
    "keep_alive_timeout": 30.0
  },
  
  "cache": {
    "enable_caching": true,
    "ttl_embeddings": 86400,
    "ttl_crawl": 7200,
    "enable_compression": true
  }
}
```

## Configuration Validation

### Pre-Deployment Validation

```bash
# Validate configuration file
python -m src.manage_config validate -c config/production.json

# Comprehensive validation with connection checks
python -m src.manage_config validate -c config/production.json --comprehensive

# Check specific environment variables
python -m src.manage_config check-env-vars

# Validate connections to external services
python -m src.manage_config check-connections
```

### Automated Validation Script

```bash
#!/bin/bash
# validate-config.sh

set -e

CONFIG_FILE=${1:-config.json}
ENVIRONMENT=${2:-production}

echo "Validating configuration for $ENVIRONMENT environment..."

# 1. Validate configuration file syntax
echo "Checking configuration syntax..."
python -m src.manage_config validate -c "$CONFIG_FILE"

# 2. Check required environment variables
echo "Checking environment variables..."
python -m src.manage_config check-env-vars

# 3. Test service connections
echo "Testing service connections..."
python -m src.manage_config check-connections

# 4. Run configuration-specific tests
echo "Running configuration tests..."
uv run pytest tests/unit/config/ -v

# 5. Generate validation report
echo "Generating validation report..."
python -m src.manage_config generate-validation-report -c "$CONFIG_FILE" -o "validation-report-$(date +%Y%m%d-%H%M%S).json"

echo "Configuration validation completed successfully!"
```

### Health Check Configuration

```json
{
  "health_checks": {
    "enabled": true,
    "interval": 30,
    "timeout": 10,
    "checks": [
      {
        "name": "qdrant",
        "type": "http",
        "url": "http://qdrant:6333/health",
        "critical": true
      },
      {
        "name": "dragonfly",
        "type": "redis",
        "url": "redis://dragonfly:6379",
        "critical": false
      },
      {
        "name": "openai",
        "type": "api",
        "endpoint": "https://api.openai.com/v1/models",
        "critical": true
      }
    ]
  }
}
```

## Change Management

### Configuration Version Control

```bash
# Track configuration changes
git add config/
git commit -m "feat(config): update production rate limits"

# Tag configuration versions
git tag -a config-v1.2.0 -m "Configuration version 1.2.0"

# Create configuration branches for major changes
git checkout -b config/memory-optimization
```

### Configuration Migration

```bash
# Migrate configuration to latest version
python -m src.manage_config migrate config.json

# Dry run to preview changes
python -m src.manage_config migrate config.json --dry-run

# Migrate to specific version
python -m src.manage_config migrate config.json --target-version 0.3.0

# Show migration path
python -m src.manage_config show-migration-path
```

### Backup and Restore

```bash
#!/bin/bash
# backup-config.sh

DATE=$(date +%Y%m%d-%H%M%S)
BACKUP_DIR="config/backups/$DATE"

mkdir -p "$BACKUP_DIR"

# Backup configuration files
cp config/*.json "$BACKUP_DIR/"
cp .env "$BACKUP_DIR/" 2>/dev/null || true

# Backup environment variables
env | grep "AI_DOCS__" > "$BACKUP_DIR/environment-variables.txt"

# Create backup archive
tar -czf "config-backup-$DATE.tar.gz" -C config/backups "$DATE"

echo "Configuration backed up to config-backup-$DATE.tar.gz"
```

### Rolling Updates

```bash
#!/bin/bash
# rolling-config-update.sh

NEW_CONFIG=${1:-config/production-new.json}
CURRENT_CONFIG=${2:-config/production.json}

echo "Performing rolling configuration update..."

# 1. Validate new configuration
python -m src.manage_config validate -c "$NEW_CONFIG"

# 2. Backup current configuration
cp "$CURRENT_CONFIG" "$CURRENT_CONFIG.backup"

# 3. Apply new configuration gradually
echo "Applying new configuration..."
cp "$NEW_CONFIG" "$CURRENT_CONFIG"

# 4. Test with new configuration
sleep 10
python -m src.manage_config check-connections

# 5. Monitor for issues
echo "Monitoring system health..."
for i in {1..6}; do
    sleep 10
    if ! python -m src.manage_config check-connections > /dev/null 2>&1; then
        echo "Health check failed, rolling back..."
        cp "$CURRENT_CONFIG.backup" "$CURRENT_CONFIG"
        exit 1
    fi
    echo "Health check $i/6 passed"
done

echo "Rolling update completed successfully!"
rm "$CURRENT_CONFIG.backup"
```

## Troubleshooting

### Common Configuration Issues

#### 1. Missing API Keys

**Symptoms:**

```text
Error: OpenAI API key required when using OpenAI embedding provider
```

**Solution:**

```bash
# Check if API key is set
echo $AI_DOCS__OPENAI__API_KEY

# Set the API key
export AI_DOCS__OPENAI__API_KEY="sk-your-api-key"

# Verify configuration
python -m src.manage_config validate -c config.json
```

#### 2. Service Connection Failures

**Symptoms:**

```text
Error: Qdrant connection failed: Connection refused
```

**Diagnosis:**

```bash
# Check service status
python -m src.manage_config check-connections

# Test individual connection
curl http://qdrant:6333/health

# Check network connectivity
ping qdrant
telnet qdrant 6333
```

**Solution:**

```bash
# Update service URL in configuration
export AI_DOCS__QDRANT__URL="http://localhost:6333"

# Or modify configuration file
{
  "qdrant": {
    "url": "http://localhost:6333"
  }
}
```

#### 3. Memory Issues

**Symptoms:**

```text
MemoryError: Unable to allocate array
```

**Diagnosis:**

```bash
# Check current memory usage
free -h
ps aux --sort=-%mem | head

# Check dispatcher stats
python -c "
from src.config import get_config
from src.services.crawling.crawl4ai_provider import Crawl4AIProvider
provider = Crawl4AIProvider(get_config().crawl4ai)
print(provider._get_dispatcher_stats())
"
```

**Solution:**

```json
{
  "crawl4ai": {
    "enable_memory_adaptive_dispatcher": true,
    "memory_threshold_percent": 60.0,
    "max_session_permit": 5,
    "enable_streaming": false
  },
  "performance": {
    "max_memory_usage_mb": 1000.0,
    "gc_threshold": 0.7
  }
}
```

#### 4. Rate Limiting Issues

**Symptoms:**

```text
RateLimitError: Rate limit exceeded
```

**Solution:**

```json
{
  "performance": {
    "default_rate_limits": {
      "openai": {
        "max_calls": 100,
        "time_window": 60
      }
    }
  },
  "crawl4ai": {
    "rate_limit_base_delay_min": 2.0,
    "rate_limit_base_delay_max": 5.0,
    "rate_limit_max_delay": 60.0,
    "rate_limit_max_retries": 3
  }
}
```

### Debug Mode Configuration

```json
{
  "debug": true,
  "log_level": "DEBUG",
  "logging": {
    "enable_file_logging": true,
    "log_file": "logs/debug.log",
    "log_rotation": "1 day",
    "log_retention": "7 days",
    "enable_performance_logging": true,
    "enable_memory_logging": true,
    "enable_configuration_logging": true
  }
}
```

### Configuration Diagnostics

```bash
# Generate comprehensive diagnostic report
python -m src.manage_config diagnose

# Check configuration completeness
python -m src.manage_config check-completeness

# Validate environment setup
python -m src.manage_config validate-environment

# Show active configuration
python -m src.manage_config show-config --format yaml
```

## Advanced Configuration Management (BJO-87)

### Configuration Management System Overview

The system includes a comprehensive configuration management framework that provides:

- **Interactive Configuration Wizard**: Guided setup with four modes (template, interactive, migration, import)
- **Configuration Templates**: Five optimized templates for different deployment scenarios
- **Backup and Restore System**: Git-like versioning with metadata and compression
- **Migration Framework**: Automated schema upgrades with rollback support
- **Enhanced Validation**: Beyond Pydantic with business rules and automatic fixes
- **CLI Management**: Rich command-line interface for all configuration operations

### Production Backup and Restore Procedures

#### 1. Configuration Backup Management

```bash
#!/bin/bash
# production-backup-procedure.sh
# Production configuration backup procedure

set -e

BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
ENVIRONMENT="${DEPLOY_ENV:-production}"
CONFIG_DIR="/app/config"
BACKUP_DIR="/backup/config"

echo "=== Configuration Backup Procedure ==="
echo "Environment: $ENVIRONMENT"
echo "Backup Date: $BACKUP_DATE"

# 1. Create backup with metadata
uv run python -m src.cli.main config backup create \
  "$CONFIG_DIR/production.json" \
  --description "Scheduled production backup - $(date)" \
  --tags "production,scheduled,backup-$BACKUP_DATE" \
  --compress \
  --incremental

# 2. Export backup for external storage
BACKUP_ID=$(uv run python -c "
from src.config.backup_restore import ConfigBackupManager
manager = ConfigBackupManager()
backups = manager.list_backups(limit=1)
print(backups[0].backup_id if backups else '')
")

if [ -n "$BACKUP_ID" ]; then
    # Export to external storage
    uv run python -m src.cli.main config backup export \
      "$BACKUP_ID" \
      --output "$BACKUP_DIR/production-backup-$BACKUP_DATE.json.gz"
    
    echo "✓ Backup exported to: $BACKUP_DIR/production-backup-$BACKUP_DATE.json.gz"
    
    # Verify backup integrity
    uv run python -m src.cli.main config backup verify "$BACKUP_ID"
    echo "✓ Backup integrity verified"
else
    echo "✗ Failed to create backup"
    exit 1
fi

# 3. Clean up old backups (keep last 30 days)
uv run python -m src.cli.main config backup cleanup \
  --config-name production \
  --keep-days 30 \
  --keep-count 50

echo "✓ Production backup completed successfully"
```

#### 2. Disaster Recovery Restore Procedure

```bash
#!/bin/bash
# disaster-recovery-restore.sh
# Production disaster recovery restore procedure

set -e

RESTORE_SOURCE=${1:-"latest"}
ENVIRONMENT=${2:-"production"}
CONFIG_PATH="/app/config/production.json"

echo "=== Disaster Recovery Configuration Restore ==="
echo "Restore Source: $RESTORE_SOURCE"
echo "Target Environment: $ENVIRONMENT"

# 1. List available backups
echo "Available backups:"
uv run python -m src.cli.main config backup list \
  --config-name production \
  --environment production \
  --limit 10

# 2. Restore from backup
if [ "$RESTORE_SOURCE" = "latest" ]; then
    # Get latest backup ID
    BACKUP_ID=$(uv run python -c "
from src.config.backup_restore import ConfigBackupManager
manager = ConfigBackupManager()
backups = manager.list_backups(config_name='production', limit=1)
print(backups[0].backup_id if backups else '')
")
else
    BACKUP_ID="$RESTORE_SOURCE"
fi

if [ -n "$BACKUP_ID" ]; then
    echo "Restoring from backup: $BACKUP_ID"
    
    # Create pre-restore backup
    uv run python -m src.cli.main config backup create \
      "$CONFIG_PATH" \
      --description "Pre-disaster-recovery backup" \
      --tags "pre-restore,disaster-recovery" \
      --force
    
    # Perform restore
    uv run python -m src.cli.main config backup restore \
      "$BACKUP_ID" \
      --target "$CONFIG_PATH" \
      --create-pre-restore-backup \
      --force
    
    echo "✓ Configuration restored from backup"
    
    # 3. Validate restored configuration
    echo "Validating restored configuration..."
    uv run python -m src.cli.main config validate \
      "$CONFIG_PATH" \
      --health-check \
      --comprehensive
    
    # 4. Test service connections
    echo "Testing service connections..."
    if uv run python -m src.cli.main config validate \
         "$CONFIG_PATH" --health-check --exit-on-error; then
        echo "✓ Disaster recovery completed successfully"
    else
        echo "✗ Service validation failed after restore"
        exit 1
    fi
else
    echo "✗ No backup found for restore"
    exit 1
fi
```

#### 3. Backup Retention and Cleanup Policies

```bash
#!/bin/bash
# backup-retention-policy.sh
# Automated backup retention and cleanup

# Production backup retention policy
PRODUCTION_KEEP_DAYS=90
PRODUCTION_KEEP_COUNT=100

# Staging backup retention policy  
STAGING_KEEP_DAYS=30
STAGING_KEEP_COUNT=50

# Development backup retention policy
DEVELOPMENT_KEEP_DAYS=7
DEVELOPMENT_KEEP_COUNT=20

echo "=== Backup Retention Policy Enforcement ==="

# Apply production retention policy
echo "Applying production backup retention policy..."
DELETED_PROD=$(uv run python -m src.cli.main config backup cleanup \
  --config-name production \
  --keep-days "$PRODUCTION_KEEP_DAYS" \
  --keep-count "$PRODUCTION_KEEP_COUNT" \
  --dry-run=false \
  --output-json | jq '.deleted_count')

echo "Production: Cleaned up $DELETED_PROD old backups"

# Apply staging retention policy
echo "Applying staging backup retention policy..."
DELETED_STAGING=$(uv run python -m src.cli.main config backup cleanup \
  --config-name staging \
  --keep-days "$STAGING_KEEP_DAYS" \
  --keep-count "$STAGING_KEEP_COUNT" \
  --dry-run=false \
  --output-json | jq '.deleted_count')

echo "Staging: Cleaned up $DELETED_STAGING old backups"

# Apply development retention policy
echo "Applying development backup retention policy..."
DELETED_DEV=$(uv run python -m src.cli.main config backup cleanup \
  --config-name development \
  --keep-days "$DEVELOPMENT_KEEP_DAYS" \
  --keep-count "$DEVELOPMENT_KEEP_COUNT" \
  --dry-run=false \
  --output-json | jq '.deleted_count')

echo "Development: Cleaned up $DELETED_DEV old backups"

# Generate retention report
cat > /tmp/backup-retention-report.json << EOF
{
  "retention_policy_run": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "policies_applied": {
    "production": {
      "keep_days": $PRODUCTION_KEEP_DAYS,
      "keep_count": $PRODUCTION_KEEP_COUNT,
      "deleted_count": $DELETED_PROD
    },
    "staging": {
      "keep_days": $STAGING_KEEP_DAYS,
      "keep_count": $STAGING_KEEP_COUNT,
      "deleted_count": $DELETED_STAGING
    },
    "development": {
      "keep_days": $DEVELOPMENT_KEEP_DAYS,
      "keep_count": $DEVELOPMENT_KEEP_COUNT,
      "deleted_count": $DELETED_DEV
    }
  },
  "total_deleted": $((DELETED_PROD + DELETED_STAGING + DELETED_DEV))
}
EOF

echo "✓ Backup retention policy enforcement completed"
echo "Report saved to: /tmp/backup-retention-report.json"
```

### Production Migration Operations

#### 1. Schema Migration Planning and Execution

```bash
#!/bin/bash
# production-migration-procedure.sh
# Production configuration schema migration procedure

set -e

TARGET_VERSION=${1:-"latest"}
CONFIG_PATH="/app/config/production.json"
ENVIRONMENT="production"

echo "=== Production Configuration Migration ==="
echo "Target Version: $TARGET_VERSION"
echo "Configuration: $CONFIG_PATH"

# 1. Pre-migration backup
echo "Creating pre-migration backup..."
PRE_MIGRATION_BACKUP=$(uv run python -m src.cli.main config backup create \
  "$CONFIG_PATH" \
  --description "Pre-migration backup for version $TARGET_VERSION" \
  --tags "pre-migration,version-$TARGET_VERSION,production" \
  --compress \
  --output-json | jq -r '.backup_id')

echo "✓ Pre-migration backup created: $PRE_MIGRATION_BACKUP"

# 2. Create and validate migration plan
echo "Creating migration plan..."
uv run python -m src.cli.main config migrate plan \
  "$CONFIG_PATH" \
  "$TARGET_VERSION" \
  --output-json > /tmp/migration-plan.json

# Validate migration plan
PLAN_VALID=$(jq '.is_valid' /tmp/migration-plan.json)
REQUIRES_DOWNTIME=$(jq '.requires_downtime' /tmp/migration-plan.json)
ESTIMATED_DURATION=$(jq -r '.estimated_duration' /tmp/migration-plan.json)

if [ "$PLAN_VALID" != "true" ]; then
    echo "✗ Migration plan validation failed"
    jq '.validation_errors[]' /tmp/migration-plan.json
    exit 1
fi

echo "✓ Migration plan validated"
echo "  Requires downtime: $REQUIRES_DOWNTIME"
echo "  Estimated duration: $ESTIMATED_DURATION"

# 3. Execute migration with monitoring
if [ "$REQUIRES_DOWNTIME" = "true" ]; then
    echo "⚠️  This migration requires downtime"
    echo "Proceeding with service maintenance window..."
    
    # Put system in maintenance mode
    # kubectl scale deployment ai-docs-api --replicas=0
fi

echo "Executing migration..."
MIGRATION_RESULT=$(uv run python -m src.cli.main config migrate apply \
  "$CONFIG_PATH" \
  "$TARGET_VERSION" \
  --force \
  --output-json)

MIGRATION_SUCCESS=$(echo "$MIGRATION_RESULT" | jq '.success')

if [ "$MIGRATION_SUCCESS" = "true" ]; then
    echo "✓ Migration completed successfully"
    
    # 4. Post-migration validation
    echo "Performing post-migration validation..."
    uv run python -m src.cli.main config validate \
      "$CONFIG_PATH" \
      --health-check \
      --comprehensive
    
    # 5. Test system functionality
    echo "Testing system functionality..."
    if ./scripts/health-check.sh; then
        echo "✓ Post-migration health checks passed"
        
        # Remove maintenance mode
        if [ "$REQUIRES_DOWNTIME" = "true" ]; then
            # kubectl scale deployment ai-docs-api --replicas=3
            echo "✓ Service restored from maintenance mode"
        fi
        
        echo "✓ Production migration completed successfully"
    else
        echo "✗ Post-migration health checks failed"
        echo "Initiating rollback procedure..."
        
        # Rollback to pre-migration backup
        uv run python -m src.cli.main config backup restore \
          "$PRE_MIGRATION_BACKUP" \
          --target "$CONFIG_PATH" \
          --force
        
        echo "✓ Rollback to pre-migration backup completed"
        exit 1
    fi
else
    echo "✗ Migration failed"
    echo "$MIGRATION_RESULT" | jq '.errors[]'
    
    # Automatic rollback
    echo "Performing automatic rollback..."
    uv run python -m src.cli.main config backup restore \
      "$PRE_MIGRATION_BACKUP" \
      --target "$CONFIG_PATH" \
      --force
    
    exit 1
fi
```

#### 2. Multi-Environment Migration Coordination

```bash
#!/bin/bash
# multi-environment-migration.sh
# Coordinate migrations across multiple environments

set -e

TARGET_VERSION=${1:-"latest"}
ENVIRONMENTS=("development" "staging" "production")

echo "=== Multi-Environment Migration Coordination ==="
echo "Target Version: $TARGET_VERSION"

# 1. Validate migration on development first
echo "Step 1: Development environment migration..."
./scripts/environment-migration.sh development "$TARGET_VERSION"

if [ $? -ne 0 ]; then
    echo "✗ Development migration failed - stopping rollout"
    exit 1
fi

echo "✓ Development migration successful"

# 2. Migrate staging environment
echo "Step 2: Staging environment migration..."
./scripts/environment-migration.sh staging "$TARGET_VERSION"

if [ $? -ne 0 ]; then
    echo "✗ Staging migration failed - stopping rollout"
    exit 1
fi

echo "✓ Staging migration successful"

# 3. Production migration with additional safeguards
echo "Step 3: Production environment migration..."
echo "⚠️  Production migration requires manual confirmation"
read -p "Proceed with production migration? (yes/no): " CONFIRM

if [ "$CONFIRM" = "yes" ]; then
    ./scripts/production-migration-procedure.sh "$TARGET_VERSION"
    
    if [ $? -eq 0 ]; then
        echo "✓ Production migration successful"
        echo "✓ Multi-environment migration rollout completed"
    else
        echo "✗ Production migration failed"
        exit 1
    fi
else
    echo "Production migration cancelled by operator"
    exit 1
fi
```

#### 3. Migration Rollback Procedures

```bash
#!/bin/bash
# migration-rollback-procedure.sh
# Production migration rollback procedure

set -e

MIGRATION_ID=${1:-"latest"}
CONFIG_PATH="/app/config/production.json"

echo "=== Migration Rollback Procedure ==="
echo "Migration ID: $MIGRATION_ID"
echo "Configuration: $CONFIG_PATH"

# 1. Identify migration to rollback
if [ "$MIGRATION_ID" = "latest" ]; then
    # Get last applied migration
    MIGRATION_ID=$(uv run python -c "
from src.config.migrations import ConfigMigrationManager
manager = ConfigMigrationManager()
history = manager.get_migration_history('$CONFIG_PATH')
print(history[-1]['migration_id'] if history else '')
")
fi

if [ -z "$MIGRATION_ID" ]; then
    echo "✗ No migration found to rollback"
    exit 1
fi

echo "Rolling back migration: $MIGRATION_ID"

# 2. Create pre-rollback backup
echo "Creating pre-rollback backup..."
PRE_ROLLBACK_BACKUP=$(uv run python -m src.cli.main config backup create \
  "$CONFIG_PATH" \
  --description "Pre-rollback backup for migration $MIGRATION_ID" \
  --tags "pre-rollback,migration-$MIGRATION_ID,production" \
  --compress \
  --output-json | jq -r '.backup_id')

echo "✓ Pre-rollback backup created: $PRE_ROLLBACK_BACKUP"

# 3. Execute rollback
echo "Executing migration rollback..."
ROLLBACK_RESULT=$(uv run python -m src.cli.main config migrate rollback \
  "$CONFIG_PATH" \
  "$MIGRATION_ID" \
  --force \
  --output-json)

ROLLBACK_SUCCESS=$(echo "$ROLLBACK_RESULT" | jq '.success')

if [ "$ROLLBACK_SUCCESS" = "true" ]; then
    echo "✓ Migration rollback completed"
    
    # 4. Validate rolled-back configuration
    echo "Validating rolled-back configuration..."
    uv run python -m src.cli.main config validate \
      "$CONFIG_PATH" \
      --health-check \
      --comprehensive
    
    # 5. Test system functionality
    echo "Testing system functionality after rollback..."
    if ./scripts/health-check.sh; then
        echo "✓ Post-rollback health checks passed"
        echo "✓ Migration rollback completed successfully"
    else
        echo "✗ Post-rollback health checks failed"
        echo "System may require manual intervention"
        exit 1
    fi
else
    echo "✗ Migration rollback failed"
    echo "$ROLLBACK_RESULT" | jq '.errors[]'
    exit 1
fi
```

### Configuration Template Management

#### 1. Template Deployment and Customization

```bash
#!/bin/bash
# template-deployment.sh
# Deploy and customize configuration templates

set -e

TEMPLATE_NAME=${1:-"production"}
ENVIRONMENT=${2:-"production"}
OUTPUT_PATH=${3:-"config/${ENVIRONMENT}.json"}

echo "=== Configuration Template Deployment ==="
echo "Template: $TEMPLATE_NAME"
echo "Environment: $ENVIRONMENT"
echo "Output: $OUTPUT_PATH"

# 1. Apply base template
echo "Applying base template..."
uv run python -m src.cli.main config template apply \
  "$TEMPLATE_NAME" \
  --output "$OUTPUT_PATH"

# 2. Apply environment-specific overrides
case "$ENVIRONMENT" in
    "production")
        echo "Applying production optimizations..."
        # High-performance settings
        export AI_DOCS__PERFORMANCE__MAX_CONCURRENT_REQUESTS=50
        export AI_DOCS__CRAWL4AI__MEMORY_THRESHOLD_PERCENT=80.0
        export AI_DOCS__CRAWL4AI__MAX_SESSION_PERMIT=50
        export AI_DOCS__CACHE__REDIS_POOL_SIZE=50
        ;;
        
    "staging")
        echo "Applying staging configurations..."
        # Moderate settings for staging
        export AI_DOCS__PERFORMANCE__MAX_CONCURRENT_REQUESTS=20
        export AI_DOCS__CRAWL4AI__MEMORY_THRESHOLD_PERCENT=75.0
        export AI_DOCS__CRAWL4AI__MAX_SESSION_PERMIT=25
        ;;
        
    "development")
        echo "Applying development configurations..."
        # Conservative settings for development
        export AI_DOCS__DEBUG=true
        export AI_DOCS__LOG_LEVEL=DEBUG
        export AI_DOCS__PERFORMANCE__MAX_CONCURRENT_REQUESTS=10
        export AI_DOCS__CRAWL4AI__MEMORY_THRESHOLD_PERCENT=70.0
        export AI_DOCS__CRAWL4AI__MAX_SESSION_PERMIT=10
        ;;
esac

# 3. Validate customized configuration
echo "Validating customized configuration..."
uv run python -m src.cli.main config validate \
  "$OUTPUT_PATH" \
  --health-check

echo "✓ Template deployment completed: $OUTPUT_PATH"
```

#### 2. Custom Template Creation

```bash
#!/bin/bash
# create-custom-template.sh
# Create custom configuration template from existing configuration

set -e

SOURCE_CONFIG=${1:-"config/production.json"}
TEMPLATE_NAME=${2:-"custom_template"}
TEMPLATE_DESC=${3:-"Custom template based on production"}

echo "=== Custom Template Creation ==="
echo "Source: $SOURCE_CONFIG"
echo "Template: $TEMPLATE_NAME"

# 1. Validate source configuration
echo "Validating source configuration..."
uv run python -m src.cli.main config validate "$SOURCE_CONFIG"

# 2. Create template from configuration
echo "Creating custom template..."
uv run python -c "
from src.config.templates import ConfigurationTemplates, ConfigurationTemplate, TemplateMetadata
from src.config.loader import UnifiedConfig
import json

# Load source configuration
with open('$SOURCE_CONFIG', 'r') as f:
    config_data = json.load(f)

# Create template metadata
metadata = TemplateMetadata(
    name='$TEMPLATE_NAME',
    description='$TEMPLATE_DESC',
    use_case='Custom template based on proven configuration',
    environment='production',
    tags=['custom', 'production-based', 'validated']
)

# Create template
template = ConfigurationTemplate(
    metadata=metadata,
    configuration=config_data,
    overrides={
        'development': {
            'debug': True,
            'log_level': 'DEBUG',
            'performance.max_concurrent_requests': 10
        },
        'staging': {
            'debug': False,
            'log_level': 'INFO',
            'performance.max_concurrent_requests': 25
        }
    }
)

# Save template
templates = ConfigurationTemplates()
templates.save_template(template, '$TEMPLATE_NAME')
print('✓ Custom template created: $TEMPLATE_NAME')
"

# 3. Test template application
echo "Testing template application..."
uv run python -m src.cli.main config template apply \
  "$TEMPLATE_NAME" \
  --output "/tmp/test-${TEMPLATE_NAME}.json"

# 4. Validate generated configuration
uv run python -m src.cli.main config validate \
  "/tmp/test-${TEMPLATE_NAME}.json"

echo "✓ Custom template validated and ready for use"
```

### Configuration Monitoring and Health Checks

#### 1. Automated Configuration Health Monitoring

```bash
#!/bin/bash
# config-health-monitor.sh
# Continuous configuration health monitoring

set -e

CONFIG_PATH=${1:-"config/production.json"}
CHECK_INTERVAL=${2:-300}  # 5 minutes
ALERT_THRESHOLD=${3:-3}   # Alert after 3 consecutive failures

FAILURE_COUNT=0

echo "=== Configuration Health Monitor Started ==="
echo "Configuration: $CONFIG_PATH"
echo "Check Interval: ${CHECK_INTERVAL}s"
echo "Alert Threshold: $ALERT_THRESHOLD failures"

while true; do
    echo "$(date): Performing configuration health check..."
    
    # Comprehensive health check
    if uv run python -m src.cli.main config validate \
         "$CONFIG_PATH" \
         --health-check \
         --comprehensive \
         --quiet; then
        
        echo "$(date): ✓ Configuration health check passed"
        FAILURE_COUNT=0
        
        # Reset any alerts
        if [ -f "/tmp/config-health-alert" ]; then
            rm "/tmp/config-health-alert"
            echo "$(date): Alert cleared - configuration healthy"
        fi
        
    else
        FAILURE_COUNT=$((FAILURE_COUNT + 1))
        echo "$(date): ✗ Configuration health check failed (attempt $FAILURE_COUNT/$ALERT_THRESHOLD)"
        
        # Trigger alert if threshold reached
        if [ $FAILURE_COUNT -ge $ALERT_THRESHOLD ]; then
            if [ ! -f "/tmp/config-health-alert" ]; then
                echo "$(date): 🚨 ALERT: Configuration health check failing repeatedly"
                
                # Generate detailed diagnostic report
                uv run python -m src.cli.main config validate \
                  "$CONFIG_PATH" \
                  --health-check \
                  --comprehensive \
                  --output-report "/tmp/config-health-report-$(date +%Y%m%d_%H%M%S).json"
                
                # Create alert marker
                touch "/tmp/config-health-alert"
                
                # Send alert (integrate with your alerting system)
                # ./scripts/send-alert.sh "Configuration health check failing" "See /tmp/config-health-report-*.json"
            fi
        fi
    fi
    
    sleep "$CHECK_INTERVAL"
done
```

#### 2. Configuration Drift Detection

```bash
#!/bin/bash
# config-drift-detection.sh
# Detect configuration drift from baseline

set -e

CURRENT_CONFIG=${1:-"config/production.json"}
BASELINE_CONFIG=${2:-"config/baseline/production.json"}

echo "=== Configuration Drift Detection ==="
echo "Current: $CURRENT_CONFIG"
echo "Baseline: $BASELINE_CONFIG"

# 1. Generate configuration hashes
CURRENT_HASH=$(uv run python -c "
from src.config.utils import ConfigVersioning
import json
with open('$CURRENT_CONFIG', 'r') as f:
    config = json.load(f)
print(ConfigVersioning.generate_config_hash(config))
")

BASELINE_HASH=$(uv run python -c "
from src.config.utils import ConfigVersioning
import json
with open('$BASELINE_CONFIG', 'r') as f:
    config = json.load(f)
print(ConfigVersioning.generate_config_hash(config))
")

# 2. Compare configurations
if [ "$CURRENT_HASH" = "$BASELINE_HASH" ]; then
    echo "✓ No configuration drift detected"
    echo "  Current hash: $CURRENT_HASH"
    echo "  Baseline hash: $BASELINE_HASH"
else
    echo "⚠️  Configuration drift detected"
    echo "  Current hash: $CURRENT_HASH"
    echo "  Baseline hash: $BASELINE_HASH"
    
    # 3. Generate detailed diff report
    echo "Generating detailed drift report..."
    uv run python -c "
import json
import difflib

# Load configurations
with open('$CURRENT_CONFIG', 'r') as f:
    current = json.load(f)
with open('$BASELINE_CONFIG', 'r') as f:
    baseline = json.load(f)

# Generate readable diff
current_str = json.dumps(current, indent=2, sort_keys=True).splitlines()
baseline_str = json.dumps(baseline, indent=2, sort_keys=True).splitlines()

diff = list(difflib.unified_diff(
    baseline_str, 
    current_str,
    fromfile='baseline',
    tofile='current',
    lineterm=''
))

if diff:
    print('Configuration Differences:')
    for line in diff:
        print(line)
else:
    print('No structural differences found')
"
    
    # 4. Check for unauthorized changes
    echo "Checking for unauthorized configuration changes..."
    # Implement your change authorization validation logic here
    
    exit 1
fi
```

## Best Practices

### 1. Environment Separation

- Use separate configuration files for each environment
- Never share production secrets with development
- Use environment-specific resource limits
- Implement proper secret management
- **Leverage configuration templates** for consistent environment setup
- **Use backup and restore** for environment synchronization

```bash
# Enhanced directory structure with advanced config management
config/
├── environments/
│   ├── production.json
│   ├── staging.json
│   ├── development.json
│   └── testing.json
├── templates/           # BJO-87: Configuration templates
│   ├── production.json
│   ├── development.json
│   ├── high_performance.json
│   ├── memory_optimized.json
│   └── distributed.json
├── backups/            # BJO-87: Configuration backups
│   ├── production/
│   ├── staging/
│   └── development/
├── migrations/         # BJO-87: Schema migrations
│   ├── v1.0.0_to_v1.1.0.py
│   ├── v1.1.0_to_v2.0.0.py
│   └── migration_registry.json
└── secrets/
    ├── production.env
    ├── staging.env
    └── development.env
```

### 2. Secret Management

```bash
# Use external secret management
export AI_DOCS__OPENAI__API_KEY=$(vault kv get -field=api_key secret/openai)
export AI_DOCS__FIRECRAWL__API_KEY=$(vault kv get -field=api_key secret/firecrawl)

# Or use Kubernetes secrets
kubectl create secret generic ai-docs-secrets \
  --from-literal=openai-api-key="sk-your-key" \
  --from-literal=firecrawl-api-key="fc-your-key"
```

### 3. Configuration Change Management

```bash
# Use configuration wizard for guided changes
uv run python -m src.cli.main config wizard \
  --config-path config/production.json

# Always create backups before changes
uv run python -m src.cli.main config backup create \
  config/production.json \
  --description "Pre-change backup - $(date)" \
  --tags "pre-change,$(whoami)"

# Validate changes before deployment
uv run python -m src.cli.main config validate \
  config/production.json \
  --health-check \
  --comprehensive
```

### 4. Migration Best Practices

```bash
# Always test migrations in development first
uv run python -m src.cli.main config migrate plan \
  config/development.json v2.0.0 \
  --dry-run

# Use staged rollouts for production migrations
./scripts/multi-environment-migration.sh v2.0.0

# Keep rollback plans ready
uv run python -m src.cli.main config migrate rollback \
  config/production.json migration_id \
  --dry-run
```

### 3. Configuration Testing

```bash
# Test configuration changes in isolation
./scripts/test-config.sh config/production-new.json

# Run configuration-specific tests
uv run pytest tests/unit/config/ -v -k "production"

# Validate with real services
python -m src.manage_config validate --comprehensive
```

### 4. Monitoring Configuration Changes

```python
# Configuration change detection
import hashlib
import json

def get_config_hash(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()

# Monitor for changes
current_hash = get_config_hash('config.json')
# Store hash and alert on changes
```

### 5. Performance Monitoring

```json
{
  "monitoring": {
    "enable_metrics": true,
    "metrics_port": 9090,
    "metrics_path": "/metrics",
    "enable_tracing": true,
    "tracing_endpoint": "http://jaeger:14268/api/traces",
    "enable_profiling": false,
    "profiling_port": 9091
  }
}
```

## Monitoring and Alerting

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'ai-docs-vector-db'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

### Key Metrics to Monitor

```python
# Configuration-related metrics
from prometheus_client import Counter, Gauge, Histogram

# Configuration metrics
config_loads = Counter('config_loads_total', 'Total configuration loads')
config_errors = Counter('config_errors_total', 'Configuration errors')
config_validation_time = Histogram('config_validation_seconds', 'Configuration validation time')

# System metrics
memory_usage = Gauge('system_memory_usage_percent', 'System memory usage percentage')
active_sessions = Gauge('crawl4ai_active_sessions', 'Active Crawl4AI sessions')
cache_hit_rate = Gauge('cache_hit_rate', 'Cache hit rate')

# Service health
service_health = Gauge('service_health', 'Service health status', ['service'])
```

### Alerting Rules

```yaml
# alerts.yml
groups:
  - name: configuration
    rules:
      - alert: ConfigurationError
        expr: increase(config_errors_total[5m]) > 0
        for: 0m
        labels:
          severity: critical
        annotations:
          summary: "Configuration error detected"
          description: "Configuration validation failed"

      - alert: HighMemoryUsage
        expr: system_memory_usage_percent > 90
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage detected"
          description: "System memory usage is above 90%"

      - alert: ServiceDown
        expr: service_health == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Service is down"
          description: "{{ $labels.service }} is not responding"
```

### Configuration Dashboard

```json
{
  "dashboard": {
    "title": "AI Docs Vector DB Configuration",
    "panels": [
      {
        "title": "Configuration Health",
        "type": "stat",
        "targets": [
          {
            "expr": "config_loads_total",
            "legendFormat": "Total Loads"
          },
          {
            "expr": "config_errors_total",
            "legendFormat": "Total Errors"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "system_memory_usage_percent",
            "legendFormat": "Memory Usage %"
          }
        ]
      },
      {
        "title": "Service Health",
        "type": "table",
        "targets": [
          {
            "expr": "service_health",
            "legendFormat": "{{ service }}"
          }
        ]
      }
    ]
  }
}
```

This comprehensive configuration guide provides operators with all the tools and knowledge needed to successfully deploy, manage, and maintain the AI Documentation Vector DB system across different environments. The guide emphasizes security, performance, and operational excellence while providing practical examples and troubleshooting guidance.
