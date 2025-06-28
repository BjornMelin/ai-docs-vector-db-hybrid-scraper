"""Intelligent Chaos Engineering Orchestrator for Automated Resilience Testing.

This module implements AI-driven chaos engineering with adaptive testing,
automated weakness detection, and intelligent experiment generation for
continuous resilience validation and improvement.
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from src.automation.self_healing.autonomous_health_monitor import AutonomousHealthMonitor, SystemMetrics
from tests.chaos.conftest import ChaosExperiment, ExperimentResult, FailureType
from tests.chaos.test_chaos_runner import ChaosTestRunner, ExperimentStatus


logger = logging.getLogger(__name__)


class WeaknessType(str, Enum):
    """Types of system weaknesses detected."""
    
    RESOURCE_CONTENTION = "resource_contention"
    DEPENDENCY_FAILURE = "dependency_failure"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    ERROR_PROPAGATION = "error_propagation"
    RECOVERY_SLOWNESS = "recovery_slowness"
    CAPACITY_LIMIT = "capacity_limit"
    CONFIGURATION_DRIFT = "configuration_drift"


class ResilienceLevel(str, Enum):
    """System resilience levels."""
    
    EXCELLENT = "excellent"
    GOOD = "good"
    ADEQUATE = "adequate"
    POOR = "poor"
    CRITICAL = "critical"


class ExperimentCategory(str, Enum):
    """Chaos experiment categories."""
    
    INFRASTRUCTURE = "infrastructure"
    APPLICATION = "application"
    NETWORK = "network"
    DATA = "data"
    SECURITY = "security"


@dataclass
class SystemWeakness:
    """Detected system weakness requiring targeted testing."""
    
    weakness_id: str
    weakness_type: WeaknessType
    component: str
    severity: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    description: str
    evidence: List[str]
    detection_time: datetime
    suggested_experiments: List[str]
    business_impact: float
    learning_objective: str
    hypothesis: str


@dataclass
class ResilienceAssessment:
    """Comprehensive system resilience assessment."""
    
    overall_resilience: ResilienceLevel
    resilience_score: float  # 0.0 to 1.0
    component_scores: Dict[str, float]
    identified_weaknesses: List[SystemWeakness]
    strengths: List[str]
    improvement_areas: List[str]
    recommended_focus_areas: List[str]
    last_assessment_time: datetime
    confidence_level: float


@dataclass
class ChaosExperimentTemplate:
    """Template for generating chaos experiments."""
    
    template_id: str
    name: str
    category: ExperimentCategory
    failure_type: FailureType
    target_weakness_types: List[WeaknessType]
    description_template: str
    parameter_ranges: Dict[str, Tuple[float, float]]
    success_criteria: List[str]
    safety_constraints: List[str]
    learning_objectives: List[str]


@dataclass
class ExperimentLearning:
    """Learning extracted from chaos experiment execution."""
    
    experiment_id: str
    weakness_addressed: Optional[str]
    hypothesis_validated: bool
    discovered_vulnerabilities: List[str]
    resilience_improvements: List[str]
    performance_insights: List[str]
    recommendations: List[str]
    confidence_score: float
    learning_timestamp: datetime


@dataclass
class ChaosResult:
    """Enhanced chaos experiment result with learning insights."""
    
    experiment_result: ExperimentResult
    recovery_analysis: Dict[str, Any]
    actual_blast_radius: Dict[str, Any]
    lessons_learned: ExperimentLearning
    resilience_impact: float
    unexpected_effects: List[str]
    system_adaptations: List[str]


class SystemWeaknessAnalyzer:
    """Analyzes system behavior to identify weaknesses for targeted chaos testing."""
    
    def __init__(self):
        self.weakness_history: List[SystemWeakness] = []
        self.component_baselines: Dict[str, Dict[str, float]] = {}
        self.pattern_database: Dict[str, List[Dict[str, Any]]] = {}
    
    async def identify_weaknesses(self, metrics: SystemMetrics) -> List[SystemWeakness]:
        """Identify system weaknesses based on current metrics and patterns."""
        weaknesses = []
        current_time = datetime.utcnow()
        
        # Analyze different types of weaknesses
        weaknesses.extend(await self._analyze_resource_contention(metrics, current_time))
        weaknesses.extend(await self._analyze_performance_degradation(metrics, current_time))
        weaknesses.extend(await self._analyze_error_patterns(metrics, current_time))
        weaknesses.extend(await self._analyze_capacity_limits(metrics, current_time))
        weaknesses.extend(await self._analyze_dependency_risks(metrics, current_time))
        
        # Filter and prioritize weaknesses
        filtered_weaknesses = await self._filter_and_prioritize_weaknesses(weaknesses)
        
        # Store in history
        self.weakness_history.extend(filtered_weaknesses)
        
        # Keep only recent history
        cutoff_time = current_time - timedelta(hours=24)
        self.weakness_history = [
            w for w in self.weakness_history if w.detection_time > cutoff_time
        ]
        
        return filtered_weaknesses
    
    async def _analyze_resource_contention(self, metrics: SystemMetrics, timestamp: datetime) -> List[SystemWeakness]:
        """Analyze for resource contention weaknesses."""
        weaknesses = []
        
        # CPU contention analysis
        if metrics.cpu_percent > 80:
            cpu_weakness = SystemWeakness(
                weakness_id=f"cpu_contention_{int(time.time())}",
                weakness_type=WeaknessType.RESOURCE_CONTENTION,
                component="cpu",
                severity=min(1.0, metrics.cpu_percent / 100),
                confidence=0.8,
                description=f"High CPU utilization ({metrics.cpu_percent:.1f}%) indicating resource contention",
                evidence=[f"CPU usage: {metrics.cpu_percent:.1f}%"],
                detection_time=timestamp,
                suggested_experiments=["cpu_stress_test", "process_termination"],
                business_impact=0.7,
                learning_objective="Validate CPU resource handling under stress",
                hypothesis="System can gracefully handle CPU resource exhaustion"
            )
            weaknesses.append(cpu_weakness)
        
        # Memory contention analysis
        if metrics.memory_percent > 85:
            memory_weakness = SystemWeakness(
                weakness_id=f"memory_contention_{int(time.time())}",
                weakness_type=WeaknessType.RESOURCE_CONTENTION,
                component="memory",
                severity=min(1.0, metrics.memory_percent / 100),
                confidence=0.9,
                description=f"High memory utilization ({metrics.memory_percent:.1f}%) indicating potential memory pressure",
                evidence=[f"Memory usage: {metrics.memory_percent:.1f}%"],
                detection_time=timestamp,
                suggested_experiments=["memory_exhaustion", "oom_killer_test"],
                business_impact=0.8,
                learning_objective="Test memory management and recovery mechanisms",
                hypothesis="System has adequate memory management and can recover from memory pressure"
            )
            weaknesses.append(memory_weakness)
        
        # Database connection contention
        if metrics.database_connections > 70:  # Assuming 100 max connections
            db_weakness = SystemWeakness(
                weakness_id=f"db_connection_contention_{int(time.time())}",
                weakness_type=WeaknessType.RESOURCE_CONTENTION,
                component="database",
                severity=min(1.0, metrics.database_connections / 100),
                confidence=0.8,
                description=f"High database connection usage ({metrics.database_connections}) may indicate connection pool issues",
                evidence=[f"DB connections: {metrics.database_connections}"],
                detection_time=timestamp,
                suggested_experiments=["connection_pool_exhaustion", "database_slowdown"],
                business_impact=0.6,
                learning_objective="Validate database connection handling and pooling",
                hypothesis="Database connection pooling is properly configured and resilient"
            )
            weaknesses.append(db_weakness)
        
        return weaknesses
    
    async def _analyze_performance_degradation(self, metrics: SystemMetrics, timestamp: datetime) -> List[SystemWeakness]:
        """Analyze for performance degradation weaknesses."""
        weaknesses = []
        
        # Response time degradation
        if metrics.response_time_p95 > 5000:  # 5 seconds
            response_weakness = SystemWeakness(
                weakness_id=f"response_degradation_{int(time.time())}",
                weakness_type=WeaknessType.PERFORMANCE_DEGRADATION,
                component="api_gateway",
                severity=min(1.0, metrics.response_time_p95 / 10000),  # Normalize to 10s max
                confidence=0.9,
                description=f"High response times ({metrics.response_time_p95:.0f}ms) indicate performance issues",
                evidence=[f"P95 response time: {metrics.response_time_p95:.0f}ms"],
                detection_time=timestamp,
                suggested_experiments=["latency_injection", "traffic_spike"],
                business_impact=0.8,
                learning_objective="Test performance under various load conditions",
                hypothesis="System maintains acceptable performance under stress"
            )
            weaknesses.append(response_weakness)
        
        # Cache performance issues
        if metrics.cache_hit_ratio < 0.7:
            cache_weakness = SystemWeakness(
                weakness_id=f"cache_performance_{int(time.time())}",
                weakness_type=WeaknessType.PERFORMANCE_DEGRADATION,
                component="cache",
                severity=1.0 - metrics.cache_hit_ratio,
                confidence=0.8,
                description=f"Low cache hit ratio ({metrics.cache_hit_ratio:.2f}) affecting performance",
                evidence=[f"Cache hit ratio: {metrics.cache_hit_ratio:.2f}"],
                detection_time=timestamp,
                suggested_experiments=["cache_invalidation", "cache_overload"],
                business_impact=0.5,
                learning_objective="Validate cache resilience and fallback mechanisms",
                hypothesis="System functions adequately when cache is compromised"
            )
            weaknesses.append(cache_weakness)
        
        return weaknesses
    
    async def _analyze_error_patterns(self, metrics: SystemMetrics, timestamp: datetime) -> List[SystemWeakness]:
        """Analyze for error propagation weaknesses."""
        weaknesses = []
        
        # High error rate
        if metrics.error_rate > 0.05:  # 5% error rate
            error_weakness = SystemWeakness(
                weakness_id=f"error_propagation_{int(time.time())}",
                weakness_type=WeaknessType.ERROR_PROPAGATION,
                component="api_gateway",
                severity=min(1.0, metrics.error_rate * 20),  # Normalize to 5% = 1.0
                confidence=0.9,
                description=f"High error rate ({metrics.error_rate:.3f}) indicates error handling issues",
                evidence=[f"Error rate: {metrics.error_rate:.3f}"],
                detection_time=timestamp,
                suggested_experiments=["error_injection", "dependency_failure"],
                business_impact=0.9,
                learning_objective="Test error handling and propagation patterns",
                hypothesis="System has robust error handling and doesn't cascade failures"
            )
            weaknesses.append(error_weakness)
        
        # Circuit breaker analysis
        open_breakers = [name for name, state in metrics.circuit_breaker_states.items() if state == 'open']
        if open_breakers:
            breaker_weakness = SystemWeakness(
                weakness_id=f"circuit_breaker_activation_{int(time.time())}",
                weakness_type=WeaknessType.DEPENDENCY_FAILURE,
                component="circuit_breakers",
                severity=min(1.0, len(open_breakers) / 5),  # Normalize to 5 breakers = 1.0
                confidence=0.8,
                description=f"Open circuit breakers ({', '.join(open_breakers)}) indicate dependency issues",
                evidence=[f"Open breakers: {', '.join(open_breakers)}"],
                detection_time=timestamp,
                suggested_experiments=["service_unavailable", "network_partition"],
                business_impact=0.7,
                learning_objective="Test circuit breaker configuration and recovery",
                hypothesis="Circuit breakers are properly configured and enable graceful degradation"
            )
            weaknesses.append(breaker_weakness)
        
        return weaknesses
    
    async def _analyze_capacity_limits(self, metrics: SystemMetrics, timestamp: datetime) -> List[SystemWeakness]:
        """Analyze for capacity limit weaknesses."""
        weaknesses = []
        
        # Disk space limits
        if metrics.disk_percent > 90:
            disk_weakness = SystemWeakness(
                weakness_id=f"disk_capacity_{int(time.time())}",
                weakness_type=WeaknessType.CAPACITY_LIMIT,
                component="storage",
                severity=min(1.0, metrics.disk_percent / 100),
                confidence=0.9,
                description=f"High disk utilization ({metrics.disk_percent:.1f}%) approaching capacity limits",
                evidence=[f"Disk usage: {metrics.disk_percent:.1f}%"],
                detection_time=timestamp,
                suggested_experiments=["disk_fill", "log_explosion"],
                business_impact=0.8,
                learning_objective="Test disk space management and cleanup mechanisms",
                hypothesis="System has adequate disk space monitoring and cleanup procedures"
            )
            weaknesses.append(disk_weakness)
        
        return weaknesses
    
    async def _analyze_dependency_risks(self, metrics: SystemMetrics, timestamp: datetime) -> List[SystemWeakness]:
        """Analyze for dependency-related weaknesses."""
        weaknesses = []
        
        # Unhealthy services
        unhealthy_services = [
            service for service, score in metrics.service_health_scores.items() 
            if score < 0.8
        ]
        
        if unhealthy_services:
            dependency_weakness = SystemWeakness(
                weakness_id=f"service_dependency_{int(time.time())}",
                weakness_type=WeaknessType.DEPENDENCY_FAILURE,
                component="service_dependencies",
                severity=min(1.0, len(unhealthy_services) / 3),  # Normalize to 3 services = 1.0
                confidence=0.8,
                description=f"Unhealthy services ({', '.join(unhealthy_services)}) indicate dependency risks",
                evidence=[f"Unhealthy services: {', '.join(unhealthy_services)}"],
                detection_time=timestamp,
                suggested_experiments=["service_dependency_failure", "cascade_failure_test"],
                business_impact=0.7,
                learning_objective="Test dependency isolation and fallback mechanisms",
                hypothesis="System gracefully handles service dependency failures"
            )
            weaknesses.append(dependency_weakness)
        
        return weaknesses
    
    async def _filter_and_prioritize_weaknesses(self, weaknesses: List[SystemWeakness]) -> List[SystemWeakness]:
        """Filter and prioritize detected weaknesses."""
        if not weaknesses:
            return []
        
        # Remove duplicates based on component and weakness type
        unique_weaknesses = {}
        for weakness in weaknesses:
            key = f"{weakness.component}_{weakness.weakness_type.value}"
            if key not in unique_weaknesses or weakness.severity > unique_weaknesses[key].severity:
                unique_weaknesses[key] = weakness
        
        # Sort by priority (severity * confidence * business_impact)
        prioritized = sorted(
            unique_weaknesses.values(),
            key=lambda w: w.severity * w.confidence * w.business_impact,
            reverse=True
        )
        
        # Return top 5 weaknesses to avoid overwhelming the system
        return prioritized[:5]


class AdaptiveChaosTestGenerator:
    """Generates targeted chaos experiments based on system weaknesses and learning objectives."""
    
    def __init__(self):
        self.experiment_templates = self._initialize_experiment_templates()
        self.experiment_history: List[ChaosExperiment] = []
        self.learning_database: Dict[str, List[ExperimentLearning]] = {}
    
    def _initialize_experiment_templates(self) -> List[ChaosExperimentTemplate]:
        """Initialize chaos experiment templates."""
        return [
            ChaosExperimentTemplate(
                template_id="cpu_stress",
                name="CPU Stress Test",
                category=ExperimentCategory.INFRASTRUCTURE,
                failure_type=FailureType.RESOURCE_EXHAUSTION,
                target_weakness_types=[WeaknessType.RESOURCE_CONTENTION],
                description_template="CPU stress test targeting {component} to validate resource handling",
                parameter_ranges={
                    "duration_seconds": (30, 300),
                    "cpu_load_percent": (80, 95),
                    "process_count": (1, 4)
                },
                success_criteria=["system_recovers", "no_data_loss", "graceful_degradation"],
                safety_constraints=["max_duration_5_minutes", "monitor_system_health"],
                learning_objectives=["CPU resource management", "Process scheduling resilience"]
            ),
            ChaosExperimentTemplate(
                template_id="memory_exhaustion",
                name="Memory Exhaustion Test",
                category=ExperimentCategory.INFRASTRUCTURE,
                failure_type=FailureType.MEMORY_EXHAUSTION,
                target_weakness_types=[WeaknessType.RESOURCE_CONTENTION],
                description_template="Memory exhaustion test for {component} to validate memory management",
                parameter_ranges={
                    "duration_seconds": (60, 240),
                    "memory_mb": (512, 2048),
                    "allocation_rate": (0.1, 0.5)
                },
                success_criteria=["system_recovers", "no_data_loss", "memory_cleanup"],
                safety_constraints=["monitor_oom_killer", "preserve_critical_processes"],
                learning_objectives=["Memory management", "OOM handling", "Garbage collection"]
            ),
            ChaosExperimentTemplate(
                template_id="network_latency",
                name="Network Latency Injection",
                category=ExperimentCategory.NETWORK,
                failure_type=FailureType.NETWORK_TIMEOUT,
                target_weakness_types=[WeaknessType.PERFORMANCE_DEGRADATION, WeaknessType.DEPENDENCY_FAILURE],
                description_template="Network latency injection for {component} dependencies",
                parameter_ranges={
                    "duration_seconds": (120, 600),
                    "latency_ms": (100, 2000),
                    "jitter_ms": (10, 100)
                },
                success_criteria=["system_recovers", "timeout_handling", "graceful_degradation"],
                safety_constraints=["preserve_health_checks", "monitor_cascade_effects"],
                learning_objectives=["Network resilience", "Timeout handling", "Dependency management"]
            ),
            ChaosExperimentTemplate(
                template_id="service_unavailable",
                name="Service Unavailable Test",
                category=ExperimentCategory.APPLICATION,
                failure_type=FailureType.SERVICE_UNAVAILABLE,
                target_weakness_types=[WeaknessType.DEPENDENCY_FAILURE, WeaknessType.ERROR_PROPAGATION],
                description_template="Service unavailable test for {component} to validate fallback mechanisms",
                parameter_ranges={
                    "duration_seconds": (60, 300),
                    "failure_rate": (0.5, 1.0),
                    "recovery_delay": (10, 60)
                },
                success_criteria=["system_recovers", "fallback_activated", "no_cascade_failure"],
                safety_constraints=["monitor_dependent_services", "preserve_data_integrity"],
                learning_objectives=["Service isolation", "Fallback mechanisms", "Error handling"]
            ),
            ChaosExperimentTemplate(
                template_id="database_slowdown",
                name="Database Performance Degradation",
                category=ExperimentCategory.DATA,
                failure_type=FailureType.SLOW_RESPONSE,
                target_weakness_types=[WeaknessType.PERFORMANCE_DEGRADATION, WeaknessType.RESOURCE_CONTENTION],
                description_template="Database slowdown simulation for {component}",
                parameter_ranges={
                    "duration_seconds": (120, 480),
                    "slowdown_factor": (2, 10),
                    "affected_queries_percent": (0.3, 0.8)
                },
                success_criteria=["system_recovers", "query_timeout_handling", "connection_pool_management"],
                safety_constraints=["preserve_data_integrity", "monitor_connection_pool"],
                learning_objectives=["Database resilience", "Query optimization", "Connection management"]
            ),
            ChaosExperimentTemplate(
                template_id="disk_fill",
                name="Disk Space Exhaustion",
                category=ExperimentCategory.INFRASTRUCTURE,
                failure_type=FailureType.DISK_FULL,
                target_weakness_types=[WeaknessType.CAPACITY_LIMIT],
                description_template="Disk space exhaustion test for {component}",
                parameter_ranges={
                    "duration_seconds": (180, 360),
                    "fill_percentage": (85, 98),
                    "fill_rate_mb_per_sec": (10, 100)
                },
                success_criteria=["system_recovers", "cleanup_mechanisms", "alert_generation"],
                safety_constraints=["preserve_critical_files", "monitor_log_rotation"],
                learning_objectives=["Disk management", "Cleanup procedures", "Monitoring effectiveness"]
            )
        ]
    
    async def generate_targeted_tests(self, weaknesses: List[SystemWeakness]) -> List[ChaosExperiment]:
        """Generate targeted chaos experiments based on identified weaknesses."""
        experiments = []
        current_time = datetime.utcnow()
        
        for weakness in weaknesses:
            # Find suitable templates for this weakness
            suitable_templates = [
                template for template in self.experiment_templates
                if weakness.weakness_type in template.target_weakness_types
            ]
            
            if not suitable_templates:
                continue
            
            # Select best template based on learning value
            best_template = await self._select_best_template(weakness, suitable_templates)
            
            # Generate experiment from template
            experiment = await self._generate_experiment_from_template(best_template, weakness)
            
            if experiment:
                experiments.append(experiment)
        
        # Add some exploratory experiments if we have capacity
        if len(experiments) < 3:
            exploratory_experiments = await self._generate_exploratory_experiments(2)
            experiments.extend(exploratory_experiments)
        
        return experiments
    
    async def _select_best_template(self, weakness: SystemWeakness, templates: List[ChaosExperimentTemplate]) -> ChaosExperimentTemplate:
        """Select the best template for addressing a specific weakness."""
        if len(templates) == 1:
            return templates[0]
        
        # Score templates based on various factors
        template_scores = {}
        
        for template in templates:
            score = 0.0
            
            # Learning value score
            learning_value = await self._calculate_learning_value(template, weakness)
            score += learning_value * 0.4
            
            # Novelty score (prefer experiments we haven't done recently)
            novelty = await self._calculate_novelty_score(template, weakness.component)
            score += novelty * 0.3
            
            # Safety score (prefer safer experiments)
            safety = len(template.safety_constraints) / 5.0  # Normalize
            score += safety * 0.2
            
            # Business impact alignment
            impact_alignment = min(1.0, weakness.business_impact)
            score += impact_alignment * 0.1
            
            template_scores[template] = score
        
        # Return highest scoring template
        return max(template_scores.items(), key=lambda x: x[1])[0]
    
    async def _generate_experiment_from_template(self, template: ChaosExperimentTemplate, weakness: SystemWeakness) -> Optional[ChaosExperiment]:
        """Generate a specific chaos experiment from a template and weakness."""
        try:
            # Customize parameters based on weakness severity
            duration_range = template.parameter_ranges.get("duration_seconds", (60, 300))
            duration = duration_range[0] + (duration_range[1] - duration_range[0]) * weakness.severity
            
            # Adjust failure rate based on confidence
            failure_rate_base = 0.5 + (0.5 * weakness.confidence)
            
            # Generate experiment
            experiment = ChaosExperiment(
                name=f"{template.name.lower().replace(' ', '_')}_{weakness.component}_{int(time.time())}",
                description=template.description_template.format(component=weakness.component),
                failure_type=template.failure_type,
                target_service=weakness.component,
                duration_seconds=int(duration),
                failure_rate=min(1.0, failure_rate_base),
                blast_radius="single",  # Start conservatively
                recovery_time_seconds=max(30, int(duration * 0.2)),
                success_criteria=template.success_criteria.copy(),
                rollback_strategy="immediate",
                metadata={
                    "template_id": template.template_id,
                    "weakness_id": weakness.weakness_id,
                    "weakness_type": weakness.weakness_type.value,
                    "learning_objective": weakness.learning_objective,
                    "hypothesis": weakness.hypothesis,
                    "severity": weakness.severity,
                    "confidence": weakness.confidence
                }
            )
            
            return experiment
            
        except Exception as e:
            logger.exception(f"Failed to generate experiment from template {template.template_id}: {e}")
            return None
    
    async def _generate_exploratory_experiments(self, count: int) -> List[ChaosExperiment]:
        """Generate exploratory experiments for discovering unknown weaknesses."""
        experiments = []
        
        # Select random templates for exploration
        available_templates = random.sample(self.experiment_templates, min(count, len(self.experiment_templates)))
        
        for template in available_templates:
            # Generate exploration experiment with moderate parameters
            experiment = ChaosExperiment(
                name=f"exploration_{template.template_id}_{int(time.time())}",
                description=f"Exploratory {template.name} to discover system behavior",
                failure_type=template.failure_type,
                target_service="system",  # General system target
                duration_seconds=random.randint(60, 180),  # Shorter for exploration
                failure_rate=random.uniform(0.3, 0.7),  # Moderate failure rate
                blast_radius="single",
                recovery_time_seconds=60,
                success_criteria=template.success_criteria.copy(),
                rollback_strategy="immediate",
                metadata={
                    "template_id": template.template_id,
                    "experiment_type": "exploratory",
                    "learning_objective": "Discover unknown system behaviors",
                    "hypothesis": "System behaves predictably under this failure mode"
                }
            )
            
            experiments.append(experiment)
        
        return experiments
    
    async def _calculate_learning_value(self, template: ChaosExperimentTemplate, weakness: SystemWeakness) -> float:
        """Calculate the learning value of applying this template to this weakness."""
        base_value = 0.5
        
        # Higher value for directly targeting the weakness type
        if weakness.weakness_type in template.target_weakness_types:
            base_value += 0.3
        
        # Higher value for high-severity weaknesses
        base_value += weakness.severity * 0.2
        
        # Check if we've learned about this combination before
        learning_key = f"{template.template_id}_{weakness.weakness_type.value}"
        if learning_key in self.learning_database:
            # Reduce value if we've tested this combination recently
            recent_learnings = [
                learning for learning in self.learning_database[learning_key]
                if (datetime.utcnow() - learning.learning_timestamp).days < 7
            ]
            if recent_learnings:
                base_value *= 0.7  # Reduce value for recent tests
        
        return min(1.0, base_value)
    
    async def _calculate_novelty_score(self, template: ChaosExperimentTemplate, component: str) -> float:
        """Calculate novelty score for this template and component combination."""
        # Check recent experiment history
        recent_experiments = [
            exp for exp in self.experiment_history[-20:]  # Last 20 experiments
            if exp.target_service == component and 
            exp.metadata.get("template_id") == template.template_id
        ]
        
        # Higher novelty for combinations we haven't tested recently
        if not recent_experiments:
            return 1.0
        elif len(recent_experiments) == 1:
            return 0.7
        elif len(recent_experiments) == 2:
            return 0.4
        else:
            return 0.1


class RecoveryValidator:
    """Validates system recovery after chaos experiments."""
    
    def __init__(self, health_monitor: AutonomousHealthMonitor):
        self.health_monitor = health_monitor
    
    async def validate_recovery(self, pre_snapshot: SystemMetrics, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Validate system recovery after chaos experiment."""
        recovery_start_time = time.time()
        
        # Wait for initial stabilization
        await asyncio.sleep(experiment.recovery_time_seconds)
        
        # Progressive recovery validation
        validation_phases = []
        max_wait_time = 300  # 5 minutes maximum
        
        while (time.time() - recovery_start_time) < max_wait_time:
            # Collect current metrics
            current_metrics = await self.health_monitor.collect_comprehensive_health_metrics()
            
            # Validate recovery phase
            phase_result = await self._validate_recovery_phase(
                pre_snapshot, current_metrics, experiment
            )
            
            validation_phases.append({
                'timestamp': time.time(),
                'recovery_time_seconds': time.time() - recovery_start_time,
                'health_recovered': phase_result['health_recovered'],
                'performance_recovered': phase_result['performance_recovered'],
                'service_health': phase_result['service_health'],
                'metrics_comparison': phase_result['metrics_comparison']
            })
            
            # Check if fully recovered
            if phase_result['fully_recovered']:
                break
            
            await asyncio.sleep(10)  # Check every 10 seconds
        
        # Generate comprehensive recovery analysis
        return {
            'total_recovery_time': time.time() - recovery_start_time,
            'recovery_successful': validation_phases[-1]['health_recovered'] if validation_phases else False,
            'recovery_progression': validation_phases,
            'performance_impact': await self._analyze_performance_impact(pre_snapshot, validation_phases),
            'resilience_score': await self._calculate_resilience_score(validation_phases),
            'lessons_learned': await self._extract_recovery_lessons(experiment, validation_phases)
        }
    
    async def _validate_recovery_phase(self, pre_snapshot: SystemMetrics, current: SystemMetrics, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Validate a single recovery phase."""
        # Health recovery validation
        health_recovered = True
        
        # Check if critical metrics are within acceptable ranges
        critical_thresholds = {
            'cpu_percent': 95,
            'memory_percent': 95,
            'disk_percent': 98,
            'error_rate': 0.1,
            'response_time_p95': 10000
        }
        
        for metric, threshold in critical_thresholds.items():
            current_value = getattr(current, metric, 0)
            if current_value > threshold:
                health_recovered = False
        
        # Performance recovery validation
        performance_recovered = True
        performance_tolerance = 1.5  # Allow 50% degradation during recovery
        
        if current.response_time_p95 > pre_snapshot.response_time_p95 * performance_tolerance:
            performance_recovered = False
        
        if current.error_rate > pre_snapshot.error_rate * 2:  # Allow double error rate during recovery
            performance_recovered = False
        
        # Service health validation
        service_health = sum(current.service_health_scores.values()) / len(current.service_health_scores) if current.service_health_scores else 0
        
        # Metrics comparison
        metrics_comparison = {
            'cpu_change': current.cpu_percent - pre_snapshot.cpu_percent,
            'memory_change': current.memory_percent - pre_snapshot.memory_percent,
            'response_time_change': current.response_time_p95 - pre_snapshot.response_time_p95,
            'error_rate_change': current.error_rate - pre_snapshot.error_rate
        }
        
        return {
            'health_recovered': health_recovered,
            'performance_recovered': performance_recovered,
            'service_health': service_health,
            'metrics_comparison': metrics_comparison,
            'fully_recovered': health_recovered and performance_recovered and service_health > 0.8
        }
    
    async def _analyze_performance_impact(self, pre_snapshot: SystemMetrics, validation_phases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance impact during recovery."""
        if not validation_phases:
            return {}
        
        # Calculate peak impact
        peak_response_time_impact = 0
        peak_error_rate_impact = 0
        
        for phase in validation_phases:
            metrics_comparison = phase['metrics_comparison']
            response_impact = metrics_comparison.get('response_time_change', 0)
            error_impact = metrics_comparison.get('error_rate_change', 0)
            
            peak_response_time_impact = max(peak_response_time_impact, response_impact)
            peak_error_rate_impact = max(peak_error_rate_impact, error_impact)
        
        # Calculate recovery slope
        if len(validation_phases) > 1:
            response_times = [phase['metrics_comparison'].get('response_time_change', 0) for phase in validation_phases]
            recovery_slope = (response_times[-1] - response_times[0]) / len(validation_phases)
        else:
            recovery_slope = 0
        
        return {
            'peak_response_time_impact_ms': peak_response_time_impact,
            'peak_error_rate_impact': peak_error_rate_impact,
            'recovery_slope': recovery_slope,
            'time_to_stable_performance': next(
                (phase['recovery_time_seconds'] for phase in validation_phases if phase['performance_recovered']),
                validation_phases[-1]['recovery_time_seconds'] if validation_phases else 0
            )
        }
    
    async def _calculate_resilience_score(self, validation_phases: List[Dict[str, Any]]) -> float:
        """Calculate resilience score based on recovery performance."""
        if not validation_phases:
            return 0.0
        
        # Base score factors
        final_phase = validation_phases[-1]
        base_score = 0.5 if final_phase['health_recovered'] else 0.0
        
        # Recovery speed bonus
        recovery_time = final_phase['recovery_time_seconds']
        if recovery_time < 60:
            base_score += 0.3  # Fast recovery
        elif recovery_time < 180:
            base_score += 0.2  # Moderate recovery
        elif recovery_time < 300:
            base_score += 0.1  # Slow recovery
        
        # Performance stability bonus
        stable_phases = sum(1 for phase in validation_phases if phase['performance_recovered'])
        stability_ratio = stable_phases / len(validation_phases)
        base_score += stability_ratio * 0.2
        
        return min(1.0, base_score)
    
    async def _extract_recovery_lessons(self, experiment: ChaosExperiment, validation_phases: List[Dict[str, Any]]) -> List[str]:
        """Extract lessons learned from recovery process."""
        lessons = []
        
        if not validation_phases:
            lessons.append("Unable to collect recovery data")
            return lessons
        
        final_phase = validation_phases[-1]
        recovery_time = final_phase['recovery_time_seconds']
        
        # Recovery time lessons
        if recovery_time < 60:
            lessons.append("System demonstrates excellent recovery speed")
        elif recovery_time > 240:
            lessons.append("Recovery time exceeds target - investigate optimization opportunities")
        
        # Performance impact lessons
        performance_phases = [phase for phase in validation_phases if not phase['performance_recovered']]
        if len(performance_phases) > len(validation_phases) * 0.5:
            lessons.append("Performance impact duration suggests need for optimization")
        
        # Health recovery lessons
        if final_phase['health_recovered']:
            lessons.append("System successfully restored healthy state")
        else:
            lessons.append("System failed to fully recover - manual intervention may be required")
        
        # Service-specific lessons
        if experiment.target_service != "system":
            lessons.append(f"{experiment.target_service} component demonstrated resilience to {experiment.failure_type.value}")
        
        return lessons


class IntelligentChaosOrchestrator:
    """AI-driven chaos engineering orchestrator with adaptive testing and learning."""
    
    def __init__(self, health_monitor: AutonomousHealthMonitor):
        self.health_monitor = health_monitor
        self.chaos_runner = ChaosTestRunner()
        self.weakness_analyzer = SystemWeaknessAnalyzer()
        self.test_generator = AdaptiveChaosTestGenerator()
        self.recovery_validator = RecoveryValidator(health_monitor)
        
        # Orchestration state
        self.orchestration_active = False
        self.experiment_history: List[ChaosResult] = []
        self.learning_database: Dict[str, List[ExperimentLearning]] = {}
        self.resilience_history: List[ResilienceAssessment] = []
        
        # Configuration
        self.testing_interval = 3600  # 1 hour between chaos sessions
        self.max_concurrent_experiments = 2
        self.safety_mode = True
        self.learning_mode = True
    
    async def start_continuous_chaos_testing(self):
        """Start continuous adaptive chaos testing."""
        if self.orchestration_active:
            logger.warning("Chaos orchestration already active")
            return
        
        self.orchestration_active = True
        logger.info("Starting intelligent chaos orchestration")
        
        try:
            await self.continuous_chaos_loop()
        except Exception as e:
            logger.exception(f"Chaos orchestration failed: {e}")
            self.orchestration_active = False
            raise
    
    async def stop_continuous_chaos_testing(self):
        """Stop continuous chaos testing."""
        self.orchestration_active = False
        logger.info("Stopped intelligent chaos orchestration")
    
    async def continuous_chaos_loop(self):
        """Main continuous chaos testing loop."""
        while self.orchestration_active:
            try:
                # 1. Assess current system resilience
                resilience_assessment = await self.assess_system_resilience()
                
                # 2. Determine if chaos testing is appropriate
                if await self.should_run_chaos_tests(resilience_assessment):
                    # 3. Generate and execute targeted experiments
                    experiment_results = await self.execute_adaptive_chaos_session(resilience_assessment)
                    
                    # 4. Learn from experiment results
                    if self.learning_mode:
                        await self.learn_from_experiments(experiment_results)
                    
                    # 5. Update resilience model
                    await self.update_resilience_model(experiment_results)
                    
                    logger.info(f"Chaos session completed: {len(experiment_results)} experiments executed")
                else:
                    logger.info("Skipping chaos testing due to system state")
                
                # 6. Generate resilience report
                await self.generate_resilience_insights()
                
            except Exception as e:
                logger.exception(f"Error in chaos orchestration loop: {e}")
            
            # Wait for next testing cycle
            await asyncio.sleep(self.testing_interval)
    
    async def assess_system_resilience(self) -> ResilienceAssessment:
        """Assess current system resilience and identify weaknesses."""
        current_time = datetime.utcnow()
        
        # Collect current system metrics
        current_metrics = await self.health_monitor.collect_comprehensive_health_metrics()
        
        # Identify system weaknesses
        weaknesses = await self.weakness_analyzer.identify_weaknesses(current_metrics)
        
        # Calculate component resilience scores
        component_scores = await self._calculate_component_resilience_scores(current_metrics, weaknesses)
        
        # Calculate overall resilience score
        overall_score = sum(component_scores.values()) / len(component_scores) if component_scores else 0.5
        
        # Determine resilience level
        resilience_level = self._determine_resilience_level(overall_score)
        
        # Identify strengths and improvement areas
        strengths = await self._identify_system_strengths(current_metrics, weaknesses)
        improvement_areas = await self._identify_improvement_areas(weaknesses)
        
        # Generate focus area recommendations
        focus_areas = await self._recommend_focus_areas(weaknesses, component_scores)
        
        assessment = ResilienceAssessment(
            overall_resilience=resilience_level,
            resilience_score=overall_score,
            component_scores=component_scores,
            identified_weaknesses=weaknesses,
            strengths=strengths,
            improvement_areas=improvement_areas,
            recommended_focus_areas=focus_areas,
            last_assessment_time=current_time,
            confidence_level=self._calculate_assessment_confidence(weaknesses)
        )
        
        # Store in history
        self.resilience_history.append(assessment)
        
        # Keep only recent history
        if len(self.resilience_history) > 100:
            self.resilience_history = self.resilience_history[-100:]
        
        return assessment
    
    async def should_run_chaos_tests(self, assessment: ResilienceAssessment) -> bool:
        """Determine if chaos tests should be executed based on system state."""
        # Don't run tests if system is already in poor condition
        if assessment.overall_resilience == ResilienceLevel.CRITICAL:
            logger.info("Skipping chaos tests: system resilience is critical")
            return False
        
        # Check if there are ongoing issues
        current_metrics = await self.health_monitor.collect_comprehensive_health_metrics()
        if (current_metrics.cpu_percent > 90 or 
            current_metrics.memory_percent > 90 or 
            current_metrics.error_rate > 0.1):
            logger.info("Skipping chaos tests: system under stress")
            return False
        
        # Check for recent experiment failures
        recent_failures = [
            result for result in self.experiment_history[-10:]
            if not result.experiment_result.success
        ]
        if len(recent_failures) > 3:
            logger.info("Skipping chaos tests: too many recent failures")
            return False
        
        # Check if we have meaningful weaknesses to test
        if not assessment.identified_weaknesses:
            logger.info("Skipping chaos tests: no significant weaknesses identified")
            return False
        
        return True
    
    async def execute_adaptive_chaos_session(self, assessment: ResilienceAssessment) -> List[ChaosResult]:
        """Execute an adaptive chaos testing session."""
        # Generate targeted experiments
        experiments = await self.test_generator.generate_targeted_tests(assessment.identified_weaknesses)
        
        if not experiments:
            logger.info("No experiments generated for this session")
            return []
        
        # Limit concurrent experiments for safety
        experiments = experiments[:self.max_concurrent_experiments]
        
        session_results = []
        
        for experiment in experiments:
            try:
                # Validate experiment safety
                if self.safety_mode and not await self._validate_experiment_safety(experiment):
                    logger.warning(f"Skipping unsafe experiment: {experiment.name}")
                    continue
                
                # Execute experiment
                result = await self.execute_chaos_experiment(experiment)
                session_results.append(result)
                
                # Brief pause between experiments
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.exception(f"Failed to execute experiment {experiment.name}: {e}")
        
        return session_results
    
    async def execute_chaos_experiment(self, experiment: ChaosExperiment) -> ChaosResult:
        """Execute a single chaos experiment with comprehensive monitoring."""
        logger.info(f"Executing chaos experiment: {experiment.name}")
        
        # Create pre-experiment system snapshot
        pre_experiment_snapshot = await self.health_monitor.collect_comprehensive_health_metrics()
        
        # Setup experiment monitoring
        monitoring_data = []
        
        async def monitoring_callback():
            metrics = await self.health_monitor.collect_comprehensive_health_metrics()
            monitoring_data.append(metrics)
            return {
                'timestamp': time.time(),
                'cpu_percent': metrics.cpu_percent,
                'memory_percent': metrics.memory_percent,
                'response_time_p95': metrics.response_time_p95,
                'error_rate': metrics.error_rate
            }
        
        try:
            # Execute experiment using chaos runner
            execution_result = await self.chaos_runner.execute_experiment(
                experiment, 
                target_system=None,  # Using default system
                monitoring_callback=monitoring_callback
            )
            
            # Validate recovery
            recovery_analysis = await self.recovery_validator.validate_recovery(
                pre_experiment_snapshot, experiment
            )
            
            # Calculate blast radius impact
            blast_radius_analysis = await self._analyze_blast_radius(
                pre_experiment_snapshot, monitoring_data, experiment
            )
            
            # Extract lessons learned
            lessons_learned = await self._extract_experiment_lessons(
                experiment, execution_result, recovery_analysis, monitoring_data
            )
            
            # Calculate resilience impact
            resilience_impact = await self._calculate_resilience_impact(
                experiment, execution_result, recovery_analysis
            )
            
            # Identify unexpected effects
            unexpected_effects = await self._identify_unexpected_effects(
                experiment, monitoring_data, pre_experiment_snapshot
            )
            
            # Identify system adaptations
            system_adaptations = await self._identify_system_adaptations(
                experiment, recovery_analysis
            )
            
            chaos_result = ChaosResult(
                experiment_result=execution_result,
                recovery_analysis=recovery_analysis,
                actual_blast_radius=blast_radius_analysis,
                lessons_learned=lessons_learned,
                resilience_impact=resilience_impact,
                unexpected_effects=unexpected_effects,
                system_adaptations=system_adaptations
            )
            
            # Store in history
            self.experiment_history.append(chaos_result)
            
            # Keep only recent history
            if len(self.experiment_history) > 500:
                self.experiment_history = self.experiment_history[-500:]
            
            return chaos_result
            
        except Exception as e:
            logger.exception(f"Chaos experiment execution failed: {e}")
            # Create failure result
            failed_result = ExperimentResult(
                experiment_name=experiment.name,
                started_at=time.time(),
                ended_at=time.time(),
                duration=0,
                failure_injected=False,
                system_recovered=False,
                recovery_time=0,
                success_criteria_met=[],
                metrics={},
                errors=[str(e)]
            )
            
            return ChaosResult(
                experiment_result=failed_result,
                recovery_analysis={},
                actual_blast_radius={},
                lessons_learned=ExperimentLearning(
                    experiment_id=experiment.name,
                    weakness_addressed=None,
                    hypothesis_validated=False,
                    discovered_vulnerabilities=[],
                    resilience_improvements=[],
                    performance_insights=[],
                    recommendations=["Review experiment safety and system state before retry"],
                    confidence_score=0.0,
                    learning_timestamp=datetime.utcnow()
                ),
                resilience_impact=0.0,
                unexpected_effects=[f"Experiment execution failed: {str(e)}"],
                system_adaptations=[]
            )
    
    async def learn_from_experiments(self, experiment_results: List[ChaosResult]):
        """Learn from experiment results and update knowledge base."""
        for result in experiment_results:
            learning = result.lessons_learned
            
            # Store learning in database
            experiment_type = result.experiment_result.experiment_name.split('_')[0]
            if experiment_type not in self.learning_database:
                self.learning_database[experiment_type] = []
            
            self.learning_database[experiment_type].append(learning)
            
            # Update test generator with learning
            self.test_generator.learning_database = self.learning_database
    
    async def _validate_experiment_safety(self, experiment: ChaosExperiment) -> bool:
        """Validate experiment safety before execution."""
        # Check blast radius
        if experiment.blast_radius == "system":
            return False
        
        # Check failure rate
        if experiment.failure_rate > 0.8:
            return False
        
        # Check duration
        if experiment.duration_seconds > 600:  # 10 minutes
            return False
        
        # Check system health
        current_metrics = await self.health_monitor.collect_comprehensive_health_metrics()
        if (current_metrics.cpu_percent > 80 or 
            current_metrics.memory_percent > 80 or 
            current_metrics.error_rate > 0.05):
            return False
        
        return True
    
    async def _calculate_component_resilience_scores(self, metrics: SystemMetrics, weaknesses: List[SystemWeakness]) -> Dict[str, float]:
        """Calculate resilience scores for each system component."""
        base_scores = {
            "cpu": 1.0 - (metrics.cpu_percent / 100),
            "memory": 1.0 - (metrics.memory_percent / 100),
            "disk": 1.0 - (metrics.disk_percent / 100),
            "api_gateway": 1.0 - min(1.0, metrics.response_time_p95 / 10000),
            "database": 1.0 - min(1.0, metrics.database_connections / 100),
            "cache": metrics.cache_hit_ratio,
            "network": 1.0 - metrics.error_rate
        }
        
        # Adjust scores based on identified weaknesses
        for weakness in weaknesses:
            if weakness.component in base_scores:
                penalty = weakness.severity * weakness.confidence * 0.3
                base_scores[weakness.component] = max(0.0, base_scores[weakness.component] - penalty)
        
        return base_scores
    
    def _determine_resilience_level(self, score: float) -> ResilienceLevel:
        """Determine resilience level based on score."""
        if score >= 0.9:
            return ResilienceLevel.EXCELLENT
        elif score >= 0.8:
            return ResilienceLevel.GOOD
        elif score >= 0.6:
            return ResilienceLevel.ADEQUATE
        elif score >= 0.4:
            return ResilienceLevel.POOR
        else:
            return ResilienceLevel.CRITICAL
    
    async def _identify_system_strengths(self, metrics: SystemMetrics, weaknesses: List[SystemWeakness]) -> List[str]:
        """Identify system strengths based on metrics and absence of weaknesses."""
        strengths = []
        
        # Performance strengths
        if metrics.response_time_p95 < 1000:
            strengths.append("Excellent response time performance")
        
        if metrics.error_rate < 0.01:
            strengths.append("Low error rate indicates robust error handling")
        
        if metrics.cache_hit_ratio > 0.9:
            strengths.append("High cache efficiency")
        
        # Resource utilization strengths
        if metrics.cpu_percent < 50:
            strengths.append("Healthy CPU utilization with good headroom")
        
        if metrics.memory_percent < 60:
            strengths.append("Efficient memory usage")
        
        # Service health strengths
        healthy_services = sum(1 for score in metrics.service_health_scores.values() if score > 0.9)
        if healthy_services > 0:
            strengths.append(f"{healthy_services} services demonstrate excellent health")
        
        # Weakness-based strengths (components not showing weaknesses)
        weak_components = {weakness.component for weakness in weaknesses}
        all_components = {"cpu", "memory", "disk", "api_gateway", "database", "cache", "network"}
        strong_components = all_components - weak_components
        
        if strong_components:
            strengths.append(f"Strong resilience in: {', '.join(strong_components)}")
        
        return strengths
    
    async def _identify_improvement_areas(self, weaknesses: List[SystemWeakness]) -> List[str]:
        """Identify improvement areas based on weaknesses."""
        improvement_areas = []
        
        # Group weaknesses by type
        weakness_types = {}
        for weakness in weaknesses:
            if weakness.weakness_type not in weakness_types:
                weakness_types[weakness.weakness_type] = []
            weakness_types[weakness.weakness_type].append(weakness)
        
        # Generate improvement recommendations
        for weakness_type, weakness_list in weakness_types.items():
            if weakness_type == WeaknessType.RESOURCE_CONTENTION:
                improvement_areas.append("Resource allocation and scaling optimization needed")
            elif weakness_type == WeaknessType.PERFORMANCE_DEGRADATION:
                improvement_areas.append("Performance monitoring and optimization required")
            elif weakness_type == WeaknessType.ERROR_PROPAGATION:
                improvement_areas.append("Error handling and circuit breaker improvements needed")
            elif weakness_type == WeaknessType.DEPENDENCY_FAILURE:
                improvement_areas.append("Dependency isolation and fallback mechanisms require attention")
            elif weakness_type == WeaknessType.CAPACITY_LIMIT:
                improvement_areas.append("Capacity planning and auto-scaling improvements needed")
        
        return improvement_areas
    
    async def _recommend_focus_areas(self, weaknesses: List[SystemWeakness], component_scores: Dict[str, float]) -> List[str]:
        """Recommend focus areas for resilience improvement."""
        focus_areas = []
        
        # Focus on highest severity weaknesses
        high_severity_weaknesses = [w for w in weaknesses if w.severity > 0.7]
        if high_severity_weaknesses:
            focus_areas.append(f"High priority: {high_severity_weaknesses[0].component} {high_severity_weaknesses[0].weakness_type.value}")
        
        # Focus on lowest scoring components
        sorted_components = sorted(component_scores.items(), key=lambda x: x[1])
        if sorted_components and sorted_components[0][1] < 0.6:
            focus_areas.append(f"Component improvement: {sorted_components[0][0]} (score: {sorted_components[0][1]:.2f})")
        
        # Focus on high business impact weaknesses
        business_critical = [w for w in weaknesses if w.business_impact > 0.8]
        if business_critical:
            focus_areas.append(f"Business critical: {business_critical[0].component} resilience")
        
        return focus_areas
    
    def _calculate_assessment_confidence(self, weaknesses: List[SystemWeakness]) -> float:
        """Calculate confidence level of the resilience assessment."""
        if not weaknesses:
            return 0.7  # Moderate confidence when no weaknesses found
        
        # Average confidence of weakness detections
        avg_confidence = sum(w.confidence for w in weaknesses) / len(weaknesses)
        
        # Adjust based on number of weaknesses (more data = higher confidence)
        sample_size_factor = min(1.0, len(weaknesses) / 5)
        
        return min(1.0, avg_confidence * 0.8 + sample_size_factor * 0.2)
    
    async def _analyze_blast_radius(self, pre_snapshot: SystemMetrics, monitoring_data: List[SystemMetrics], experiment: ChaosExperiment) -> Dict[str, Any]:
        """Analyze the actual blast radius of the experiment."""
        # Compare metrics during experiment to baseline
        if not monitoring_data:
            return {"analysis_incomplete": True}
        
        # Calculate impact on different components
        impact_analysis = {}
        
        for component in ["cpu", "memory", "response_time", "error_rate"]:
            baseline = getattr(pre_snapshot, f"{component}_percent" if component in ["cpu", "memory"] else component, 0)
            
            # Find peak impact during experiment
            peak_impact = baseline
            for metrics in monitoring_data:
                current_value = getattr(metrics, f"{component}_percent" if component in ["cpu", "memory"] else component, 0)
                if component in ["cpu", "memory", "response_time", "error_rate"]:
                    peak_impact = max(peak_impact, current_value)
            
            impact_percentage = ((peak_impact - baseline) / baseline * 100) if baseline > 0 else 0
            impact_analysis[component] = {
                "baseline": baseline,
                "peak": peak_impact,
                "impact_percentage": impact_percentage
            }
        
        return {
            "target_component": experiment.target_service,
            "intended_blast_radius": experiment.blast_radius,
            "component_impacts": impact_analysis,
            "cascading_effects": len([comp for comp, data in impact_analysis.items() if data["impact_percentage"] > 10]),
            "contained_impact": all(data["impact_percentage"] < 50 for data in impact_analysis.values())
        }
    
    async def _extract_experiment_lessons(self, experiment: ChaosExperiment, execution_result, recovery_analysis: Dict[str, Any], monitoring_data: List[SystemMetrics]) -> ExperimentLearning:
        """Extract comprehensive lessons from experiment execution."""
        # Validate hypothesis
        hypothesis_validated = execution_result.success and recovery_analysis.get('recovery_successful', False)
        
        # Discover vulnerabilities
        vulnerabilities = []
        if not execution_result.success:
            vulnerabilities.append(f"System failed to handle {experiment.failure_type.value} in {experiment.target_service}")
        
        if recovery_analysis.get('total_recovery_time', 0) > 180:
            vulnerabilities.append("Slow recovery time indicates optimization opportunity")
        
        # Identify resilience improvements
        improvements = []
        if execution_result.success:
            improvements.append(f"Confirmed resilience to {experiment.failure_type.value}")
        
        if recovery_analysis.get('resilience_score', 0) > 0.8:
            improvements.append("Demonstrated excellent recovery capabilities")
        
        # Performance insights
        insights = []
        peak_impact = max(
            (getattr(m, 'response_time_p95', 0) for m in monitoring_data),
            default=0
        )
        if peak_impact > 0:
            insights.append(f"Peak response time impact: {peak_impact:.0f}ms")
        
        # Generate recommendations
        recommendations = recovery_analysis.get('lessons_learned', [])
        
        return ExperimentLearning(
            experiment_id=experiment.name,
            weakness_addressed=experiment.metadata.get('weakness_id'),
            hypothesis_validated=hypothesis_validated,
            discovered_vulnerabilities=vulnerabilities,
            resilience_improvements=improvements,
            performance_insights=insights,
            recommendations=recommendations,
            confidence_score=min(1.0, execution_result.success + recovery_analysis.get('resilience_score', 0) / 2),
            learning_timestamp=datetime.utcnow()
        )
    
    async def _calculate_resilience_impact(self, experiment: ChaosExperiment, execution_result, recovery_analysis: Dict[str, Any]) -> float:
        """Calculate the impact of this experiment on overall system resilience understanding."""
        base_impact = 0.5
        
        # Successful experiments provide more insight
        if execution_result.success:
            base_impact += 0.3
        
        # Recovery validation adds insight
        if recovery_analysis.get('recovery_successful', False):
            base_impact += 0.2
        
        # Target specific weaknesses provide more learning
        if experiment.metadata.get('weakness_id'):
            base_impact += 0.2
        
        # Novel experiments provide more insight
        similar_experiments = [
            exp for exp in self.experiment_history[-20:]
            if exp.experiment_result.experiment_name.startswith(experiment.name.split('_')[0])
        ]
        if len(similar_experiments) < 2:
            base_impact += 0.1
        
        return min(1.0, base_impact)
    
    async def _identify_unexpected_effects(self, experiment: ChaosExperiment, monitoring_data: List[SystemMetrics], pre_snapshot: SystemMetrics) -> List[str]:
        """Identify unexpected effects during experiment execution."""
        unexpected = []
        
        if not monitoring_data:
            return unexpected
        
        # Check for unexpected resource spikes
        for metrics in monitoring_data:
            if experiment.target_service != "cpu" and metrics.cpu_percent > pre_snapshot.cpu_percent + 30:
                unexpected.append(f"Unexpected CPU spike to {metrics.cpu_percent:.1f}%")
            
            if experiment.target_service != "memory" and metrics.memory_percent > pre_snapshot.memory_percent + 20:
                unexpected.append(f"Unexpected memory increase to {metrics.memory_percent:.1f}%")
            
            if experiment.failure_type != FailureType.NETWORK_TIMEOUT and metrics.response_time_p95 > pre_snapshot.response_time_p95 * 3:
                unexpected.append(f"Unexpected response time spike to {metrics.response_time_p95:.0f}ms")
        
        return list(set(unexpected))  # Remove duplicates
    
    async def _identify_system_adaptations(self, experiment: ChaosExperiment, recovery_analysis: Dict[str, Any]) -> List[str]:
        """Identify system adaptations observed during the experiment."""
        adaptations = []
        
        # Check recovery patterns
        if recovery_analysis.get('recovery_successful', False):
            recovery_time = recovery_analysis.get('total_recovery_time', 0)
            if recovery_time < 60:
                adaptations.append("System demonstrated rapid self-recovery")
            elif recovery_time < 180:
                adaptations.append("System recovered within acceptable time frame")
        
        # Check resilience mechanisms
        resilience_score = recovery_analysis.get('resilience_score', 0)
        if resilience_score > 0.8:
            adaptations.append("Strong resilience mechanisms activated successfully")
        
        return adaptations
    
    async def update_resilience_model(self, experiment_results: List[ChaosResult]):
        """Update the resilience model based on experiment results."""
        # This would update ML models in a full implementation
        # For now, we'll update the weakness analyzer patterns
        for result in experiment_results:
            if result.experiment_result.success:
                # Record successful patterns
                experiment_type = result.experiment_result.experiment_name.split('_')[0]
                weakness_type = result.lessons_learned.weakness_addressed
                
                if weakness_type:
                    pattern_key = f"{experiment_type}_{weakness_type}"
                    if pattern_key not in self.weakness_analyzer.pattern_database:
                        self.weakness_analyzer.pattern_database[pattern_key] = []
                    
                    self.weakness_analyzer.pattern_database[pattern_key].append({
                        "success": True,
                        "resilience_impact": result.resilience_impact,
                        "recovery_time": result.recovery_analysis.get('total_recovery_time', 0),
                        "timestamp": datetime.utcnow()
                    })
    
    async def generate_resilience_insights(self):
        """Generate insights about system resilience based on accumulated data."""
        if not self.resilience_history:
            return
        
        latest_assessment = self.resilience_history[-1]
        
        # Log resilience status
        logger.info(f"=== Resilience Status ===")
        logger.info(f"Overall Resilience: {latest_assessment.overall_resilience.value}")
        logger.info(f"Resilience Score: {latest_assessment.resilience_score:.2f}")
        logger.info(f"Identified Weaknesses: {len(latest_assessment.identified_weaknesses)}")
        logger.info(f"Component Scores: {latest_assessment.component_scores}")
        logger.info(f"Focus Areas: {latest_assessment.recommended_focus_areas}")
        logger.info(f"=== End Resilience Status ===")
        
        # Log recent experiment summary
        recent_experiments = self.experiment_history[-10:]
        if recent_experiments:
            successful = sum(1 for exp in recent_experiments if exp.experiment_result.success)
            logger.info(f"Recent Chaos Testing: {successful}/{len(recent_experiments)} experiments successful")
    
    async def get_orchestration_status(self) -> Dict[str, Any]:
        """Get current orchestration status and statistics."""
        recent_experiments = self.experiment_history[-20:] if self.experiment_history else []
        recent_successful = sum(1 for exp in recent_experiments if exp.experiment_result.success)
        
        return {
            "orchestration_active": self.orchestration_active,
            "testing_interval_seconds": self.testing_interval,
            "max_concurrent_experiments": self.max_concurrent_experiments,
            "safety_mode": self.safety_mode,
            "learning_mode": self.learning_mode,
            "total_experiments_executed": len(self.experiment_history),
            "recent_success_rate": recent_successful / len(recent_experiments) if recent_experiments else 0,
            "resilience_assessments_count": len(self.resilience_history),
            "current_resilience_level": self.resilience_history[-1].overall_resilience.value if self.resilience_history else "unknown",
            "learning_database_size": sum(len(learnings) for learnings in self.learning_database.values()),
            "active_weakness_patterns": len(self.weakness_analyzer.pattern_database)
        }