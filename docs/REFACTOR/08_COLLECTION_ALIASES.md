# Collection Aliases Implementation Guide

**GitHub Issue**: [#62](https://github.com/BjornMelin/ai-docs-vector-db-hybrid-scraper/issues/62)

## Overview

Collection aliases in Qdrant enable zero-downtime updates, A/B testing, and seamless rollbacks. This is crucial for production systems where search availability must be maintained during updates.

## Why Collection Aliases?

### Current Problem

- Updating embeddings requires rebuilding entire collection
- Search downtime during reindexing
- No easy rollback if issues arise
- Can't test changes safely in production

### Solution with Aliases

- Build new collection in background
- Instant switch with alias update
- Zero downtime deployment
- Easy rollback to previous version
- A/B testing capabilities

## Implementation

### 1. Alias Manager

```python
# src/services/qdrant_alias_manager.py
from typing import Optional, Any
import asyncio
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import (
    CreateAliasOperation,
    DeleteAliasOperation,
    AliasOperations,
)

from .logging_config import get_logger

logger = get_logger(__name__)

class QdrantAliasManager:
    """Manage Qdrant collection aliases for zero-downtime updates."""
    
    def __init__(self, client: QdrantClient):
        self.client = client
        
    async def create_alias(
        self,
        alias_name: str,
        collection_name: str,
        force: bool = False
    ) -> bool:
        """Create or update an alias to point to a collection."""
        
        try:
            # Check if alias exists
            if await self.alias_exists(alias_name):
                if not force:
                    logger.warning(f"Alias {alias_name} already exists")
                    return False
                
                # Delete existing alias
                await self.delete_alias(alias_name)
            
            # Create new alias
            self.client.update_aliases(
                AliasOperations(
                    actions=[
                        CreateAliasOperation(
                            create_alias={
                                "alias_name": alias_name,
                                "collection_name": collection_name,
                            }
                        )
                    ]
                )
            )
            
            logger.info(f"Created alias {alias_name} -> {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create alias: {e}")
            raise
    
    async def switch_alias(
        self,
        alias_name: str,
        new_collection: str,
        delete_old: bool = False
    ) -> Optional[str]:
        """Atomically switch alias to new collection."""
        
        try:
            # Get current collection
            old_collection = await self.get_collection_for_alias(alias_name)
            
            if old_collection == new_collection:
                logger.warning("Alias already points to target collection")
                return None
            
            # Atomic switch
            operations = []
            
            # Delete old alias
            if old_collection:
                operations.append(
                    DeleteAliasOperation(
                        delete_alias={"alias_name": alias_name}
                    )
                )
            
            # Create new alias
            operations.append(
                CreateAliasOperation(
                    create_alias={
                        "alias_name": alias_name,
                        "collection_name": new_collection,
                    }
                )
            )
            
            # Execute atomically
            self.client.update_aliases(
                AliasOperations(actions=operations)
            )
            
            logger.info(f"Switched alias {alias_name}: {old_collection} -> {new_collection}")
            
            # Optionally delete old collection
            if delete_old and old_collection:
                await self.safe_delete_collection(old_collection)
            
            return old_collection
            
        except Exception as e:
            logger.error(f"Failed to switch alias: {e}")
            raise
    
    async def delete_alias(self, alias_name: str) -> bool:
        """Delete an alias."""
        
        try:
            self.client.update_aliases(
                AliasOperations(
                    actions=[
                        DeleteAliasOperation(
                            delete_alias={"alias_name": alias_name}
                        )
                    ]
                )
            )
            
            logger.info(f"Deleted alias {alias_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete alias: {e}")
            return False
    
    async def alias_exists(self, alias_name: str) -> bool:
        """Check if an alias exists."""
        
        try:
            aliases = self.client.get_aliases()
            return any(alias.alias_name == alias_name for alias in aliases.aliases)
        except:
            return False
    
    async def get_collection_for_alias(self, alias_name: str) -> Optional[str]:
        """Get collection name that alias points to."""
        
        try:
            aliases = self.client.get_aliases()
            for alias in aliases.aliases:
                if alias.alias_name == alias_name:
                    return alias.collection_name
            return None
        except:
            return None
    
    async def list_aliases(self) -> dict[str, str]:
        """List all aliases and their collections."""
        
        try:
            aliases = self.client.get_aliases()
            return {
                alias.alias_name: alias.collection_name
                for alias in aliases.aliases
            }
        except Exception as e:
            logger.error(f"Failed to list aliases: {e}")
            return {}
    
    async def safe_delete_collection(
        self,
        collection_name: str,
        grace_period_minutes: int = 60
    ):
        """Safely delete collection after grace period."""
        
        # Check if any alias points to this collection
        aliases = await self.list_aliases()
        if collection_name in aliases.values():
            logger.warning(f"Collection {collection_name} still has aliases")
            return
        
        # Schedule deletion after grace period
        logger.info(f"Scheduling deletion of {collection_name} in {grace_period_minutes} minutes")
        
        await asyncio.sleep(grace_period_minutes * 60)
        
        # Double-check no aliases
        aliases = await self.list_aliases()
        if collection_name not in aliases.values():
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection {collection_name}")
```

### 2. Blue-Green Deployment Pattern

```python
# src/services/deployment/blue_green.py
class BlueGreenDeployment:
    """Implement blue-green deployment for vector collections."""
    
    def __init__(self, qdrant_service: QdrantService, alias_manager: QdrantAliasManager):
        self.qdrant = qdrant_service
        self.aliases = alias_manager
        
    async def deploy_new_version(
        self,
        alias_name: str,
        data_source: str,
        validation_queries: list[str],
        rollback_on_failure: bool = True
    ) -> dict[str, Any]:
        """Deploy new collection version with validation."""
        
        # Get current collection (blue)
        blue_collection = await self.aliases.get_collection_for_alias(alias_name)
        if not blue_collection:
            raise ValueError(f"No collection found for alias {alias_name}")
        
        # Create new collection (green)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        green_collection = f"{alias_name}_{timestamp}"
        
        logger.info(f"Creating green collection: {green_collection}")
        
        try:
            # 1. Create new collection with same config
            await self.qdrant.clone_collection_schema(
                source=blue_collection,
                target=green_collection
            )
            
            # 2. Populate new collection
            await self._populate_collection(green_collection, data_source)
            
            # 3. Validate new collection
            validation_passed = await self._validate_collection(
                green_collection,
                validation_queries
            )
            
            if not validation_passed:
                raise ValueError("Validation failed for new collection")
            
            # 4. Switch alias atomically
            await self.aliases.switch_alias(
                alias_name=alias_name,
                new_collection=green_collection
            )
            
            # 5. Monitor for issues
            await self._monitor_after_switch(alias_name, duration_seconds=300)
            
            # 6. Schedule old collection cleanup via task queue
            # This ensures cleanup survives server restarts
            await self.aliases.safe_delete_collection(blue_collection)
            
            return {
                "success": True,
                "old_collection": blue_collection,
                "new_collection": green_collection,
                "alias": alias_name,
            }
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            
            if rollback_on_failure:
                await self._rollback(alias_name, blue_collection, green_collection)
            
            raise
    
    async def _populate_collection(self, collection_name: str, data_source: str):
        """Populate collection from data source."""
        
        # Implementation depends on data source
        # Could be from backup, fresh crawl, or migration
        logger.info(f"Populating {collection_name} from {data_source}")
        
        # Example: Copy from existing collection
        if data_source.startswith("collection:"):
            source_collection = data_source.replace("collection:", "")
            await self.qdrant.copy_collection_data(
                source=source_collection,
                target=collection_name
            )
    
    async def _validate_collection(
        self,
        collection_name: str,
        validation_queries: list[str]
    ) -> bool:
        """Validate collection with test queries."""
        
        logger.info(f"Validating {collection_name}")
        
        for query in validation_queries:
            try:
                # Generate embedding for query
                embedding = await generate_embedding(query)
                
                # Search in new collection
                results = await self.qdrant.search(
                    collection_name=collection_name,
                    query_vector=embedding,
                    limit=10
                )
                
                # Validate results
                if not results or len(results) < 5:
                    logger.error(f"Validation failed for query: {query}")
                    return False
                
                # Check result quality
                if results[0].score < 0.7:
                    logger.error(f"Low score for query: {query}")
                    return False
                    
            except Exception as e:
                logger.error(f"Validation error: {e}")
                return False
        
        logger.info("All validations passed")
        return True
    
    async def _monitor_after_switch(self, alias_name: str, duration_seconds: int):
        """Monitor collection after switch for issues."""
        
        logger.info(f"Monitoring {alias_name} for {duration_seconds}s")
        
        start_time = asyncio.get_event_loop().time()
        error_count = 0
        
        while asyncio.get_event_loop().time() - start_time < duration_seconds:
            try:
                # Perform health check
                collection = await self.aliases.get_collection_for_alias(alias_name)
                info = self.qdrant.client.get_collection(collection)
                
                if info.status != "green":
                    error_count += 1
                    logger.warning(f"Collection status: {info.status}")
                
                if error_count > 5:
                    raise ValueError("Too many errors after switch")
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                error_count += 1
            
            await asyncio.sleep(10)
    
    async def _rollback(
        self,
        alias_name: str,
        old_collection: str,
        new_collection: str
    ):
        """Rollback to previous collection."""
        
        logger.warning(f"Rolling back {alias_name} to {old_collection}")
        
        try:
            # Switch alias back
            await self.aliases.switch_alias(
                alias_name=alias_name,
                new_collection=old_collection
            )
            
            # Delete failed collection
            self.qdrant.client.delete_collection(new_collection)
            
            logger.info("Rollback completed")
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
```

### 3. A/B Testing Implementation

```python
# src/services/deployment/ab_testing.py
class ABTestingManager:
    """Manage A/B testing for vector search."""
    
    def __init__(self, qdrant_service: QdrantService):
        self.qdrant = qdrant_service
        self.experiments = {}
    
    async def create_experiment(
        self,
        experiment_name: str,
        control_collection: str,
        treatment_collection: str,
        traffic_split: float = 0.5,
        metrics_to_track: list[str] = None
    ) -> str:
        """Create A/B test experiment."""
        
        experiment_id = f"exp_{experiment_name}_{int(time.time())}"
        
        self.experiments[experiment_id] = {
            "name": experiment_name,
            "control": control_collection,
            "treatment": treatment_collection,
            "traffic_split": traffic_split,
            "metrics": metrics_to_track or ["latency", "relevance", "clicks"],
            "results": {
                "control": defaultdict(list),
                "treatment": defaultdict(list),
            },
            "start_time": time.time(),
        }
        
        logger.info(f"Created experiment {experiment_id}")
        return experiment_id
    
    async def route_query(
        self,
        experiment_id: str,
        query_vector: list[float],
        user_id: Optional[str] = None
    ) -> tuple[str, list[Any]]:
        """Route query to control or treatment based on experiment."""
        
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Deterministic routing based on user_id or random
        if user_id:
            # Hash user_id for consistent routing
            variant = "treatment" if hash(user_id) % 100 < experiment["traffic_split"] * 100 else "control"
        else:
            # Random assignment
            variant = "treatment" if random.random() < experiment["traffic_split"] else "control"
        
        # Execute search
        collection = experiment[variant]
        start_time = time.time()
        
        results = await self.qdrant.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=10
        )
        
        # Track metrics
        latency = time.time() - start_time
        experiment["results"][variant]["latency"].append(latency)
        
        return variant, results
    
    async def track_feedback(
        self,
        experiment_id: str,
        variant: str,
        metric: str,
        value: float
    ):
        """Track user feedback for experiment."""
        
        experiment = self.experiments.get(experiment_id)
        if experiment and metric in experiment["metrics"]:
            experiment["results"][variant][metric].append(value)
    
    def analyze_experiment(self, experiment_id: str) -> dict[str, Any]:
        """Analyze A/B test results."""
        
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            return {}
        
        results = {}
        
        for metric in experiment["metrics"]:
            control_data = experiment["results"]["control"][metric]
            treatment_data = experiment["results"]["treatment"][metric]
            
            if control_data and treatment_data:
                # Calculate statistics
                control_mean = np.mean(control_data)
                treatment_mean = np.mean(treatment_data)
                
                # Perform t-test
                from scipy import stats
                t_stat, p_value = stats.ttest_ind(control_data, treatment_data)
                
                results[metric] = {
                    "control_mean": control_mean,
                    "treatment_mean": treatment_mean,
                    "improvement": (treatment_mean - control_mean) / control_mean,
                    "p_value": p_value,
                    "significant": p_value < 0.05,
                }
        
        return results
```

### 4. Canary Deployment

```python
# src/services/deployment/canary.py
class CanaryDeployment:
    """Gradual rollout of new collections."""
    
    def __init__(self, alias_manager: QdrantAliasManager):
        self.aliases = alias_manager
        self.deployments = {}
    
    async def start_canary(
        self,
        alias_name: str,
        new_collection: str,
        stages: list[dict] = None
    ) -> str:
        """Start canary deployment with gradual traffic shift."""
        
        if not stages:
            # Default canary stages
            stages = [
                {"percentage": 5, "duration_minutes": 30},
                {"percentage": 25, "duration_minutes": 60},
                {"percentage": 50, "duration_minutes": 120},
                {"percentage": 100, "duration_minutes": 0},
            ]
        
        deployment_id = f"canary_{int(time.time())}"
        
        old_collection = await self.aliases.get_collection_for_alias(alias_name)
        
        self.deployments[deployment_id] = {
            "alias": alias_name,
            "old_collection": old_collection,
            "new_collection": new_collection,
            "stages": stages,
            "current_stage": 0,
            "metrics": defaultdict(list),
        }
        
        # Start canary process via task queue for persistence
        await self.canary.start_canary(deployment_config, auto_rollback=True)
        
        return deployment_id
    
    async def _run_canary(self, deployment_id: str):
        """Execute canary deployment stages."""
        
        deployment = self.deployments[deployment_id]
        
        for i, stage in enumerate(deployment["stages"]):
            deployment["current_stage"] = i
            percentage = stage["percentage"]
            duration = stage["duration_minutes"]
            
            logger.info(f"Canary stage {i}: {percentage}% traffic for {duration} minutes")
            
            # Monitor during this stage
            if duration > 0:
                await self._monitor_stage(deployment_id, duration * 60)
            
            # Check metrics before proceeding
            if not self._check_health(deployment_id):
                logger.error("Canary failed health check, rolling back")
                await self._rollback_canary(deployment_id)
                return
            
            # If 100%, complete the deployment
            if percentage == 100:
                await self.aliases.switch_alias(
                    alias_name=deployment["alias"],
                    new_collection=deployment["new_collection"]
                )
                logger.info("Canary deployment completed successfully")
                return
    
    async def _monitor_stage(self, deployment_id: str, duration_seconds: int):
        """Monitor metrics during canary stage."""
        
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            # Collect metrics
            metrics = await self._collect_metrics(deployment_id)
            self.deployments[deployment_id]["metrics"]["latency"].append(metrics["latency"])
            self.deployments[deployment_id]["metrics"]["error_rate"].append(metrics["error_rate"])
            
            await asyncio.sleep(60)  # Check every minute
    
    def _check_health(self, deployment_id: str) -> bool:
        """Check if canary is healthy."""
        
        metrics = self.deployments[deployment_id]["metrics"]
        
        if not metrics["error_rate"]:
            return True
        
        # Check error rate
        recent_errors = metrics["error_rate"][-10:]
        if np.mean(recent_errors) > 0.05:  # 5% error threshold
            return False
        
        # Check latency
        recent_latency = metrics["latency"][-10:]
        if np.mean(recent_latency) > 200:  # 200ms threshold
            return False
        
        return True
```

### 5. Integration with MCP Server

```python
# Update MCP server to use aliases
@server.tool()
async def search_with_alias(
    query: str,
    alias: str = "documentation",
    limit: int = 10
) -> list[SearchResult]:
    """Search using collection alias for zero-downtime updates."""
    
    # Get actual collection from alias
    collection = await alias_manager.get_collection_for_alias(alias)
    if not collection:
        raise ValueError(f"Alias {alias} not found")
    
    # Perform search on actual collection
    return await qdrant_service.search(
        collection_name=collection,
        query_text=query,
        limit=limit
    )

@server.tool()
async def deploy_new_index(
    alias: str,
    source: str,
    validate: bool = True
) -> dict:
    """Deploy new index version with zero downtime."""
    
    deployment = BlueGreenDeployment(qdrant_service, alias_manager)
    
    validation_queries = [
        "python asyncio",
        "react hooks",
        "fastapi authentication",
    ] if validate else []
    
    result = await deployment.deploy_new_version(
        alias_name=alias,
        data_source=source,
        validation_queries=validation_queries
    )
    
    return result
```

## Usage Examples

### Basic Alias Operations

```python
# Create initial alias
await alias_manager.create_alias(
    alias_name="documentation",
    collection_name="docs_v1"
)

# Deploy new version
await alias_manager.switch_alias(
    alias_name="documentation",
    new_collection="docs_v2",
    delete_old=True  # Clean up after grace period
)
```

### Blue-Green Deployment

```python
# Deploy new documentation index
result = await blue_green.deploy_new_version(
    alias_name="documentation",
    data_source="collection:docs_latest",
    validation_queries=[
        "python tutorial",
        "javascript async await",
        "docker compose",
    ]
)
```

### A/B Testing

```python
# Create experiment
exp_id = await ab_testing.create_experiment(
    experiment_name="new_embeddings",
    control_collection="docs_v1",
    treatment_collection="docs_v2_experimental",
    traffic_split=0.2  # 20% to treatment
)

# Route queries through experiment
variant, results = await ab_testing.route_query(exp_id, query_vector)

# Analyze after sufficient data
analysis = ab_testing.analyze_experiment(exp_id)
```

## Benefits

1. **Zero Downtime**: Switch instantly between collections
2. **Safe Rollback**: Keep old version until confident
3. **A/B Testing**: Test changes on real traffic
4. **Gradual Rollout**: Canary deployments for safety
5. **Atomic Updates**: No partial state during switch

## Next Steps

1. Implement automated index rebuilding schedule
2. Add collection versioning metadata
3. Create deployment dashboard
4. Set up alerting for failed deployments
5. Build collection backup/restore system
