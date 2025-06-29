# Dynamic Tool Discovery System - Technical Specifications

## Overview

The Dynamic Tool Discovery System enables runtime detection, registration, and management of MCP tools without requiring server restarts. This system provides the foundation for a truly extensible and scalable MCP server architecture.

## Architecture Components

### 1. Discovery Engine Core

```python
# src/mcp_tools/discovery/engine.py

import asyncio
import inspect
import logging
from typing import Dict, List, Optional, Set, Type, Union, Any
from pathlib import Path
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import importlib.util
import hashlib
import json

logger = logging.getLogger(__name__)

@dataclass
class ToolInstance:
    """Runtime tool instance with metadata."""
    metadata: 'ToolMetadata'
    callable: Any
    registration_time: float
    last_used: float
    usage_count: int = 0
    performance_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class DiscoveryResult:
    """Result of tool discovery operation."""
    discovered_tools: List['ToolMetadata']
    failed_discoveries: List[Dict[str, str]]
    discovery_time: float
    source_info: Dict[str, Any]

class ToolDiscoveryEngine:
    """Core engine for dynamic tool discovery and management."""
    
    def __init__(self, config: 'DiscoveryConfig'):
        self.config = config
        self.discovered_tools: Dict[str, ToolMetadata] = {}
        self.active_tools: Dict[str, ToolInstance] = {}
        self.discovery_plugins: Dict[str, 'DiscoveryPlugin'] = {}
        self.discovery_cache: Dict[str, DiscoveryResult] = {}
        self.tool_dependencies: Dict[str, Set[str]] = {}
        self.watchers: List['ToolWatcher'] = []
        self._discovery_lock = asyncio.Lock()
        
    async def initialize(self) -> None:
        """Initialize the discovery engine."""
        logger.info("Initializing tool discovery engine")
        
        # Load discovery plugins
        await self._load_discovery_plugins()
        
        # Start file system watchers
        await self._start_watchers()
        
        # Perform initial discovery
        await self.discover_all_tools()
        
        logger.info(f"Discovery engine initialized with {len(self.active_tools)} tools")
    
    async def discover_all_tools(self) -> DiscoveryResult:
        """Discover tools from all configured sources."""
        async with self._discovery_lock:
            start_time = time.time()
            all_discovered = []
            all_failed = []
            
            # Discover from each plugin
            for plugin_name, plugin in self.discovery_plugins.items():
                try:
                    result = await plugin.discover_tools()
                    all_discovered.extend(result.discovered_tools)
                    all_failed.extend(result.failed_discoveries)
                    logger.info(f"Plugin {plugin_name} discovered {len(result.discovered_tools)} tools")
                except Exception as e:
                    logger.error(f"Plugin {plugin_name} discovery failed: {e}")
                    all_failed.append({
                        "plugin": plugin_name,
                        "error": str(e),
                        "type": "plugin_failure"
                    })
            
            # Update internal state
            for tool_metadata in all_discovered:
                self.discovered_tools[tool_metadata.id] = tool_metadata
            
            discovery_time = time.time() - start_time
            result = DiscoveryResult(
                discovered_tools=all_discovered,
                failed_discoveries=all_failed,
                discovery_time=discovery_time,
                source_info={"plugins_used": list(self.discovery_plugins.keys())}
            )
            
            # Cache result
            cache_key = self._generate_discovery_cache_key()
            self.discovery_cache[cache_key] = result
            
            return result
    
    async def register_tool_dynamically(self, 
                                      tool_metadata: 'ToolMetadata',
                                      mcp_server: 'FastMCP') -> bool:
        """Register a discovered tool with the MCP server at runtime."""
        try:
            # Validate tool metadata
            validation_result = await self._validate_tool_metadata(tool_metadata)
            if not validation_result.is_valid:
                logger.error(f"Tool validation failed: {validation_result.errors}")
                return False
            
            # Check dependencies
            if not await self._check_tool_dependencies(tool_metadata):
                logger.error(f"Tool dependencies not satisfied: {tool_metadata.id}")
                return False
            
            # Load tool implementation
            tool_callable = await self._load_tool_implementation(tool_metadata)
            if not tool_callable:
                logger.error(f"Failed to load tool implementation: {tool_metadata.id}")
                return False
            
            # Register with FastMCP
            registration_success = await self._register_with_fastmcp(
                tool_metadata, tool_callable, mcp_server
            )
            
            if registration_success:
                # Create tool instance
                tool_instance = ToolInstance(
                    metadata=tool_metadata,
                    callable=tool_callable,
                    registration_time=time.time(),
                    last_used=0.0
                )
                
                self.active_tools[tool_metadata.id] = tool_instance
                logger.info(f"Successfully registered tool: {tool_metadata.id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to register tool {tool_metadata.id}: {e}")
            return False
    
    async def unregister_tool(self, tool_id: str) -> bool:
        """Safely unregister a tool from the MCP server."""
        try:
            if tool_id not in self.active_tools:
                logger.warning(f"Tool not found for unregistration: {tool_id}")
                return False
            
            tool_instance = self.active_tools[tool_id]
            
            # Check if tool is in use
            if await self._is_tool_in_use(tool_id):
                logger.warning(f"Cannot unregister tool in use: {tool_id}")
                return False
            
            # Unregister from FastMCP
            # Note: FastMCP doesn't have built-in unregistration, so we need to work around this
            await self._unregister_from_fastmcp(tool_id)
            
            # Remove from active tools
            del self.active_tools[tool_id]
            
            logger.info(f"Successfully unregistered tool: {tool_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister tool {tool_id}: {e}")
            return False
    
    async def get_tool_capabilities(self, tool_id: str) -> Optional['ToolCapabilities']:
        """Get detailed capabilities for a specific tool."""
        if tool_id not in self.active_tools:
            return None
        
        tool_instance = self.active_tools[tool_id]
        metadata = tool_instance.metadata
        
        return ToolCapabilities(
            tool_id=tool_id,
            name=metadata.name,
            version=metadata.version,
            capabilities=metadata.capabilities,
            input_schema=metadata.input_schema,
            output_schema=metadata.output_schema,
            performance_profile=metadata.performance_profile,
            runtime_metrics=tool_instance.performance_metrics,
            dependencies=metadata.dependencies,
            security_requirements=metadata.security_requirements
        )
    
    async def search_tools(self, 
                          query: 'ToolQuery') -> List['ToolMetadata']:
        """Search for tools based on capabilities and requirements."""
        matching_tools = []
        
        for tool_metadata in self.discovered_tools.values():
            if await self._matches_query(tool_metadata, query):
                matching_tools.append(tool_metadata)
        
        # Sort by relevance score
        matching_tools.sort(key=lambda t: self._calculate_relevance_score(t, query), reverse=True)
        
        return matching_tools
    
    async def get_discovery_status(self) -> 'DiscoveryStatus':
        """Get current discovery system status."""
        return DiscoveryStatus(
            total_discovered=len(self.discovered_tools),
            active_tools=len(self.active_tools),
            discovery_plugins=list(self.discovery_plugins.keys()),
            last_discovery_time=max(
                (result.discovery_time for result in self.discovery_cache.values()),
                default=0.0
            ),
            watchers_active=len([w for w in self.watchers if w.is_active]),
            cache_size=len(self.discovery_cache)
        )
    
    # Private methods
    
    async def _load_discovery_plugins(self) -> None:
        """Load and initialize discovery plugins."""
        plugin_classes = [
            FileSystemDiscoveryPlugin,
            NetworkDiscoveryPlugin,
            DatabaseDiscoveryPlugin,
            GitRepositoryDiscoveryPlugin
        ]
        
        for plugin_class in plugin_classes:
            try:
                plugin_config = self.config.get_plugin_config(plugin_class.__name__)
                if plugin_config.enabled:
                    plugin = plugin_class(plugin_config)
                    await plugin.initialize()
                    self.discovery_plugins[plugin_class.__name__] = plugin
                    logger.info(f"Loaded discovery plugin: {plugin_class.__name__}")
            except Exception as e:
                logger.error(f"Failed to load plugin {plugin_class.__name__}: {e}")
    
    async def _start_watchers(self) -> None:
        """Start file system and network watchers."""
        for watch_config in self.config.watch_configs:
            try:
                watcher = ToolWatcher(watch_config, self)
                await watcher.start()
                self.watchers.append(watcher)
                logger.info(f"Started watcher for: {watch_config.path}")
            except Exception as e:
                logger.error(f"Failed to start watcher for {watch_config.path}: {e}")
    
    async def _validate_tool_metadata(self, 
                                    metadata: 'ToolMetadata') -> 'ValidationResult':
        """Validate tool metadata before registration."""
        errors = []
        warnings = []
        
        # Basic validation
        if not metadata.id or not metadata.name:
            errors.append("Tool ID and name are required")
        
        # Schema validation
        try:
            # Validate input/output schemas are valid JSON schemas
            import jsonschema
            jsonschema.Draft7Validator.check_schema(metadata.input_schema)
            jsonschema.Draft7Validator.check_schema(metadata.output_schema)
        except Exception as e:
            errors.append(f"Invalid schema: {e}")
        
        # Capability validation
        if not metadata.capabilities:
            warnings.append("No capabilities specified")
        
        # Security validation
        if metadata.capabilities and ToolCapability.NETWORK in metadata.capabilities:
            if not metadata.security_requirements.network_access_approved:
                errors.append("Network access not approved for network-capable tool")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    async def _check_tool_dependencies(self, metadata: 'ToolMetadata') -> bool:
        """Check if tool dependencies are satisfied."""
        for dep_id in metadata.dependencies:
            if dep_id not in self.active_tools:
                logger.warning(f"Dependency not satisfied: {dep_id}")
                return False
        return True
    
    async def _load_tool_implementation(self, 
                                      metadata: 'ToolMetadata') -> Optional[Any]:
        """Load the actual tool implementation."""
        try:
            if metadata.source_type == "file":
                return await self._load_from_file(metadata.source_path)
            elif metadata.source_type == "module":
                return await self._load_from_module(metadata.module_path)
            elif metadata.source_type == "network":
                return await self._load_from_network(metadata.network_endpoint)
            else:
                logger.error(f"Unknown source type: {metadata.source_type}")
                return None
        except Exception as e:
            logger.error(f"Failed to load tool implementation: {e}")
            return None
    
    async def _register_with_fastmcp(self, 
                                   metadata: 'ToolMetadata',
                                   tool_callable: Any,
                                   mcp_server: 'FastMCP') -> bool:
        """Register tool with FastMCP server."""
        try:
            # Create wrapper function with proper signature
            async def tool_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await tool_callable(*args, **kwargs)
                    
                    # Update metrics
                    tool_instance = self.active_tools.get(metadata.id)
                    if tool_instance:
                        tool_instance.last_used = time.time()
                        tool_instance.usage_count += 1
                        execution_time = time.time() - start_time
                        tool_instance.performance_metrics["avg_execution_time"] = (
                            (tool_instance.performance_metrics.get("avg_execution_time", 0) * 
                             (tool_instance.usage_count - 1) + execution_time) / 
                            tool_instance.usage_count
                        )
                    
                    return result
                except Exception as e:
                    logger.error(f"Tool execution error for {metadata.id}: {e}")
                    raise
            
            # Register with FastMCP using decorator approach
            decorated_tool = mcp_server.tool(name=metadata.id, 
                                           description=metadata.description)(tool_wrapper)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to register with FastMCP: {e}")
            return False
    
    def _generate_discovery_cache_key(self) -> str:
        """Generate cache key for discovery results."""
        key_data = {
            "plugins": sorted(self.discovery_plugins.keys()),
            "config_hash": self.config.get_hash(),
            "timestamp": int(time.time() / 300)  # 5-minute buckets
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    def _matches_query(self, metadata: 'ToolMetadata', query: 'ToolQuery') -> bool:
        """Check if tool metadata matches search query."""
        # Capability matching
        if query.required_capabilities:
            if not all(cap in metadata.capabilities for cap in query.required_capabilities):
                return False
        
        # Text search in name/description
        if query.text_search:
            search_text = query.text_search.lower()
            if search_text not in metadata.name.lower() and search_text not in metadata.description.lower():
                return False
        
        # Version constraints
        if query.version_constraints:
            if not self._version_matches(metadata.version, query.version_constraints):
                return False
        
        return True
    
    def _calculate_relevance_score(self, metadata: 'ToolMetadata', query: 'ToolQuery') -> float:
        """Calculate relevance score for search results."""
        score = 0.0
        
        # Exact name match bonus
        if query.text_search and query.text_search.lower() == metadata.name.lower():
            score += 10.0
        
        # Capability match bonus
        if query.required_capabilities:
            matching_caps = len(set(query.required_capabilities) & set(metadata.capabilities))
            score += matching_caps * 2.0
        
        # Usage popularity bonus
        tool_instance = self.active_tools.get(metadata.id)
        if tool_instance:
            score += min(tool_instance.usage_count / 100.0, 5.0)
        
        # Performance bonus
        if metadata.performance_profile.avg_execution_time_ms < 100:
            score += 3.0
        elif metadata.performance_profile.avg_execution_time_ms < 500:
            score += 1.0
        
        return score
```

### 2. Discovery Plugins

```python
# src/mcp_tools/discovery/plugins/filesystem.py

class FileSystemDiscoveryPlugin(DiscoveryPlugin):
    """Discover tools from filesystem paths."""
    
    def __init__(self, config: 'FileSystemDiscoveryConfig'):
        self.config = config
        self.supported_extensions = ['.py', '.js', '.ts', '.json']
        
    async def discover_tools(self) -> DiscoveryResult:
        """Discover tools from configured filesystem paths."""
        start_time = time.time()
        discovered_tools = []
        failed_discoveries = []
        
        for search_path in self.config.search_paths:
            try:
                tools = await self._scan_directory(Path(search_path))
                discovered_tools.extend(tools)
            except Exception as e:
                failed_discoveries.append({
                    "path": str(search_path),
                    "error": str(e),
                    "type": "filesystem_scan_error"
                })
        
        return DiscoveryResult(
            discovered_tools=discovered_tools,
            failed_discoveries=failed_discoveries,
            discovery_time=time.time() - start_time,
            source_info={"search_paths": [str(p) for p in self.config.search_paths]}
        )
    
    async def _scan_directory(self, directory: Path) -> List['ToolMetadata']:
        """Scan directory for tool definitions."""
        tools = []
        
        for file_path in directory.rglob("*"):
            if file_path.suffix in self.supported_extensions:
                try:
                    tool_metadata = await self._extract_tool_metadata(file_path)
                    if tool_metadata:
                        tools.append(tool_metadata)
                except Exception as e:
                    logger.warning(f"Failed to extract metadata from {file_path}: {e}")
        
        return tools
    
    async def _extract_tool_metadata(self, file_path: Path) -> Optional['ToolMetadata']:
        """Extract tool metadata from file."""
        if file_path.suffix == '.py':
            return await self._extract_from_python(file_path)
        elif file_path.suffix == '.json':
            return await self._extract_from_json(file_path)
        # Add more extractors as needed
        return None
```

### 3. Tool Metadata Models

```python
# src/mcp_tools/discovery/models.py

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from datetime import datetime

class ToolCapability(str, Enum):
    """Tool capability types."""
    READ = "read"
    WRITE = "write"
    COMPUTE = "compute"
    NETWORK = "network"
    DATABASE = "database"
    FILE_SYSTEM = "file_system"
    STREAMING = "streaming"
    BATCH = "batch"

class ToolSourceType(str, Enum):
    """Tool source types."""
    FILE = "file"
    MODULE = "module"
    NETWORK = "network"
    DATABASE = "database"
    GIT = "git"

class PerformanceProfile(BaseModel):
    """Tool performance characteristics."""
    avg_execution_time_ms: float = Field(description="Average execution time in milliseconds")
    max_execution_time_ms: float = Field(description="Maximum execution time in milliseconds")
    memory_usage_mb: float = Field(description="Average memory usage in MB")
    cpu_intensive: bool = Field(default=False, description="Whether tool is CPU intensive")
    io_intensive: bool = Field(default=False, description="Whether tool is I/O intensive")
    parallelizable: bool = Field(default=True, description="Whether tool can run in parallel")

class SecurityRequirements(BaseModel):
    """Tool security requirements."""
    network_access_approved: bool = Field(default=False)
    file_system_access_required: bool = Field(default=False)
    database_access_required: bool = Field(default=False)
    elevated_privileges_required: bool = Field(default=False)
    sandbox_compatible: bool = Field(default=True)
    encryption_required: bool = Field(default=False)

class ToolMetadata(BaseModel):
    """Comprehensive tool metadata."""
    
    # Basic identification
    id: str = Field(description="Unique tool identifier")
    name: str = Field(description="Human-readable tool name")
    version: str = Field(description="Tool version")
    description: str = Field(description="Tool description")
    
    # Capabilities and requirements
    capabilities: List[ToolCapability] = Field(description="Tool capabilities")
    dependencies: List[str] = Field(default_factory=list, description="Required tool dependencies")
    
    # Schema definitions
    input_schema: Dict[str, Any] = Field(description="JSON schema for input validation")
    output_schema: Dict[str, Any] = Field(description="JSON schema for output validation")
    
    # Performance and security
    performance_profile: PerformanceProfile = Field(description="Performance characteristics")
    security_requirements: SecurityRequirements = Field(description="Security requirements")
    
    # Source information
    source_type: ToolSourceType = Field(description="How to load the tool")
    source_path: Optional[str] = Field(default=None, description="File path for file-based tools")
    module_path: Optional[str] = Field(default=None, description="Module path for module-based tools")
    network_endpoint: Optional[str] = Field(default=None, description="Network endpoint for remote tools")
    
    # Metadata
    author: Optional[str] = Field(default=None, description="Tool author")
    license: Optional[str] = Field(default=None, description="Tool license")
    tags: List[str] = Field(default_factory=list, description="Searchable tags")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    @validator('id')
    def validate_id(cls, v):
        """Validate tool ID format."""
        if not v or not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError("Tool ID must be alphanumeric with underscores or hyphens")
        return v
    
    @validator('version')
    def validate_version(cls, v):
        """Validate semantic version format."""
        import re
        if not re.match(r'^\d+\.\d+\.\d+(-\w+)?$', v):
            raise ValueError("Version must follow semantic versioning (e.g., 1.0.0)")
        return v

class ToolCapabilities(BaseModel):
    """Runtime tool capabilities and status."""
    tool_id: str
    name: str
    version: str
    capabilities: List[ToolCapability]
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    performance_profile: PerformanceProfile
    runtime_metrics: Dict[str, float]
    dependencies: List[str]
    security_requirements: SecurityRequirements
    is_active: bool = True
    last_health_check: Optional[datetime] = None

class ToolQuery(BaseModel):
    """Tool search query."""
    text_search: Optional[str] = Field(default=None, description="Text search in name/description")
    required_capabilities: Optional[List[ToolCapability]] = Field(default=None)
    version_constraints: Optional[str] = Field(default=None, description="Version constraints (e.g., >=1.0.0)")
    performance_requirements: Optional[PerformanceProfile] = Field(default=None)
    security_constraints: Optional[SecurityRequirements] = Field(default=None)
    tags: Optional[List[str]] = Field(default=None, description="Required tags")

class DiscoveryStatus(BaseModel):
    """Discovery system status."""
    total_discovered: int
    active_tools: int
    discovery_plugins: List[str]
    last_discovery_time: float
    watchers_active: int
    cache_size: int
    health_status: str = "healthy"

class ValidationResult(BaseModel):
    """Tool validation result."""
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
```

### 4. Configuration

```python
# src/mcp_tools/discovery/config.py

class DiscoveryConfig(BaseModel):
    """Configuration for tool discovery system."""
    
    # Discovery settings
    auto_discovery_enabled: bool = Field(default=True)
    discovery_interval_seconds: int = Field(default=300, description="Auto-discovery interval")
    max_concurrent_discoveries: int = Field(default=5)
    
    # Cache settings
    cache_discovery_results: bool = Field(default=True)
    cache_ttl_seconds: int = Field(default=3600)
    max_cache_entries: int = Field(default=1000)
    
    # Plugin configurations
    filesystem_discovery: 'FileSystemDiscoveryConfig' = Field(default_factory=lambda: FileSystemDiscoveryConfig())
    network_discovery: 'NetworkDiscoveryConfig' = Field(default_factory=lambda: NetworkDiscoveryConfig())
    database_discovery: 'DatabaseDiscoveryConfig' = Field(default_factory=lambda: DatabaseDiscoveryConfig())
    git_discovery: 'GitRepositoryDiscoveryConfig' = Field(default_factory=lambda: GitRepositoryDiscoveryConfig())
    
    # Watcher configurations
    watch_configs: List['WatchConfig'] = Field(default_factory=list)
    
    # Security settings
    allow_unsigned_tools: bool = Field(default=False)
    require_security_approval: bool = Field(default=True)
    sandbox_untrusted_tools: bool = Field(default=True)
    
    def get_plugin_config(self, plugin_name: str) -> 'PluginConfig':
        """Get configuration for specific plugin."""
        return getattr(self, plugin_name.lower().replace('plugin', ''))
    
    def get_hash(self) -> str:
        """Get configuration hash for caching."""
        return hashlib.md5(self.json().encode()).hexdigest()
```

## Integration Points

### 1. FastMCP Integration

The discovery system integrates with FastMCP through a wrapper that:
- Intercepts tool registrations
- Adds metadata tracking
- Enables runtime registration/unregistration
- Provides usage metrics

### 2. Performance Monitoring Integration

Tools discovered dynamically are automatically instrumented with:
- Execution time tracking
- Memory usage monitoring
- Error rate tracking
- Usage pattern analysis

### 3. Security Integration

All discovered tools go through security validation:
- Capability-based access control
- Code signing verification (optional)
- Sandbox execution for untrusted tools
- Audit logging for all tool operations

## Benefits

1. **Zero-Downtime Tool Updates**: Add/remove tools without server restart
2. **Automatic Tool Discovery**: Automatically find and register new tools
3. **Enhanced Developer Experience**: Simplified tool development and deployment
4. **Better Resource Management**: Track and optimize tool usage
5. **Improved Security**: Comprehensive validation and sandboxing

This dynamic discovery system provides the foundation for a truly extensible MCP server that can adapt to changing requirements without manual intervention.