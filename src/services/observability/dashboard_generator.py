"""Advanced dashboard generation with portfolio-quality visualizations.

This module creates sophisticated monitoring dashboards inspired by DataDog and New Relic,
showcasing enterprise-grade observability patterns and visual design principles.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class VisualizationType(Enum):
    """Professional visualization types for monitoring dashboards."""

    SINGLE_STAT = "singlestat"
    TIME_SERIES = "graph"
    HEAT_MAP = "heatmap"
    TABLE = "table"
    PIE_CHART = "piechart"
    BAR_GAUGE = "bargauge"
    GAUGE = "gauge"
    LOGS = "logs"
    ALERT_LIST = "alertlist"
    TEXT = "text"
    NODE_GRAPH = "nodeGraph"


class ThresholdLevel(Enum):
    """Threshold levels for alerting and visualization."""

    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


@dataclass
class Threshold:
    """Performance threshold configuration."""

    value: float
    level: ThresholdLevel
    operator: str = "gt"  # gt, gte, lt, lte, eq


@dataclass
class VisualizationConfig:
    """Configuration for individual dashboard visualizations."""

    title: str
    type: VisualizationType
    query: str
    description: str | None = None
    unit: str = "short"
    thresholds: List[Threshold] = None
    width: int = 12
    height: int = 8
    x_pos: int = 0
    y_pos: int = 0
    legend_visible: bool = True
    tooltip_shared: bool = True

    def __post_init__(self):
        if self.thresholds is None:
            self.thresholds = []


class DashboardTheme(BaseModel):
    """Professional dashboard theming configuration."""

    primary_color: str = "#1f77b4"
    secondary_color: str = "#ff7f0e"
    success_color: str = "#2ca02c"
    warning_color: str = "#ff7f0e"
    danger_color: str = "#d62728"
    background_color: str = "#fafafa"
    panel_background: str = "#ffffff"
    text_color: str = "#212529"

    def get_color_palette(self) -> List[str]:
        """Get a comprehensive color palette for visualizations."""
        return [
            self.primary_color,
            self.secondary_color,
            self.success_color,
            self.warning_color,
            self.danger_color,
            "#9467bd",  # Purple
            "#8c564b",  # Brown
            "#e377c2",  # Pink
            "#7f7f7f",  # Gray
            "#bcbd22",  # Olive
            "#17becf",  # Cyan
        ]


class DashboardTemplate(BaseModel):
    """Template for generating professional monitoring dashboards."""

    name: str
    description: str
    category: str = "AI/ML Monitoring"
    tags: List[str] = Field(default_factory=list)
    refresh_interval: str = "5s"
    time_range_from: str = "now-1h"
    time_range_to: str = "now"
    theme: DashboardTheme = Field(default_factory=DashboardTheme)
    variables: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    visualizations: List[VisualizationConfig] = Field(default_factory=list)

    def add_variable(
        self, name: str, query: str, label: str | None = None, multi: bool = False
    ):
        """Add a template variable for dynamic dashboard filtering."""
        self.variables[name] = {
            "name": name,
            "label": label or name.title(),
            "query": query,
            "multi": multi,
            "includeAll": multi,
            "refresh": "time",
            "type": "query",
        }

    def add_visualization(self, config: VisualizationConfig):
        """Add a visualization to the dashboard."""
        self.visualizations.append(config)


class ProfessionalDashboardGenerator:
    """Generate portfolio-quality monitoring dashboards with enterprise UX patterns.

    Features:
    - DataDog/New Relic inspired visual design
    - Responsive layout with automatic positioning
    - Advanced color schemes and theming
    - Context-aware metric selection
    - Performance optimization indicators
    - Cost attribution visualizations
    - Anomaly detection displays
    """

    def __init__(self, theme: DashboardTheme | None = None):
        self.theme = theme or DashboardTheme()
        self.templates: Dict[str, DashboardTemplate] = {}
        self._load_default_templates()

    def _load_default_templates(self):
        """Load default professional dashboard templates."""

        # Executive Overview Dashboard
        exec_template = DashboardTemplate(
            name="Executive Overview",
            description="High-level system health and business metrics for executives",
            tags=["executive", "overview", "kpi"],
            refresh_interval="30s",
            time_range_from="now-24h",
        )

        # Add executive-focused visualizations
        exec_template.add_visualization(
            VisualizationConfig(
                title="System Health Score",
                type=VisualizationType.GAUGE,
                query="avg(up{job='ai-docs-scraper'}) * 100",
                description="Overall system availability percentage",
                unit="percent",
                thresholds=[
                    Threshold(95.0, ThresholdLevel.GREEN),
                    Threshold(90.0, ThresholdLevel.YELLOW),
                    Threshold(0.0, ThresholdLevel.RED),
                ],
                width=6,
                height=6,
            )
        )

        exec_template.add_visualization(
            VisualizationConfig(
                title="Daily AI Operations Cost",
                type=VisualizationType.SINGLE_STAT,
                query="sum(increase(ai_operation_cost_usd_total[24h]))",
                description="Total AI service costs for the last 24 hours",
                unit="currencyUSD",
                width=6,
                height=6,
                x_pos=6,
            )
        )

        exec_template.add_visualization(
            VisualizationConfig(
                title="Request Volume Trend",
                type=VisualizationType.TIME_SERIES,
                query="sum(rate(http_requests_total[5m]))",
                description="Requests per second over time",
                unit="reqps",
                width=12,
                height=8,
                y_pos=6,
            )
        )

        self.templates["executive"] = exec_template

        # Technical Deep Dive Dashboard
        tech_template = DashboardTemplate(
            name="Technical Deep Dive",
            description="Detailed technical metrics for engineers and operators",
            tags=["technical", "performance", "debugging"],
            refresh_interval="5s",
        )

        # Add variables for filtering
        tech_template.add_variable("instance", "label_values(up, instance)", "Instance")
        tech_template.add_variable("job", "label_values(up, job)", "Service")

        # Performance visualizations
        tech_template.add_visualization(
            VisualizationConfig(
                title="Response Time Percentiles",
                type=VisualizationType.TIME_SERIES,
                query="histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{instance='$instance'}[5m]))",
                description="P95, P50 response times for monitoring performance degradation",
                unit="s",
                width=12,
                height=8,
            )
        )

        tech_template.add_visualization(
            VisualizationConfig(
                title="Error Rate by Endpoint",
                type=VisualizationType.TIME_SERIES,
                query="sum by (endpoint) (rate(http_requests_total{status=~'5..', instance='$instance'}[5m]))",
                description="5xx error rate broken down by API endpoint",
                unit="reqps",
                thresholds=[
                    Threshold(0.0, ThresholdLevel.GREEN),
                    Threshold(0.01, ThresholdLevel.YELLOW),
                    Threshold(0.05, ThresholdLevel.RED),
                ],
                width=12,
                height=8,
                y_pos=8,
            )
        )

        tech_template.add_visualization(
            VisualizationConfig(
                title="Memory Usage Heatmap",
                type=VisualizationType.HEAT_MAP,
                query="process_resident_memory_bytes{instance='$instance'}",
                description="Memory usage patterns across time and instances",
                unit="bytes",
                width=12,
                height=6,
                y_pos=16,
            )
        )

        self.templates["technical"] = tech_template

        # AI/ML Operations Dashboard
        ai_template = DashboardTemplate(
            name="AI/ML Operations",
            description="AI-specific metrics including costs, performance, and quality",
            tags=["ai", "ml", "cost-optimization"],
            refresh_interval="10s",
        )

        ai_template.add_visualization(
            VisualizationConfig(
                title="Embedding Generation Rate",
                type=VisualizationType.TIME_SERIES,
                query="sum(rate(ai_embeddings_generated_total[5m]))",
                description="Rate of embedding generation across all providers",
                unit="ops",
                width=6,
                height=8,
            )
        )

        ai_template.add_visualization(
            VisualizationConfig(
                title="Cost by AI Provider",
                type=VisualizationType.PIE_CHART,
                query="sum by (provider) (ai_operation_cost_usd_total)",
                description="Cost breakdown by AI service provider",
                unit="currencyUSD",
                width=6,
                height=8,
                x_pos=6,
            )
        )

        ai_template.add_visualization(
            VisualizationConfig(
                title="Model Performance Comparison",
                type=VisualizationType.BAR_GAUGE,
                query="avg by (model) (ai_operation_duration_seconds)",
                description="Average latency by AI model for performance comparison",
                unit="s",
                width=12,
                height=6,
                y_pos=8,
            )
        )

        ai_template.add_visualization(
            VisualizationConfig(
                title="Token Usage Efficiency",
                type=VisualizationType.TIME_SERIES,
                query="rate(ai_tokens_processed_total[5m]) / rate(ai_operation_cost_usd_total[5m])",
                description="Tokens processed per dollar spent - efficiency metric",
                unit="tokens/USD",
                width=12,
                height=8,
                y_pos=14,
            )
        )

        self.templates["ai_operations"] = ai_template

        # Security & Compliance Dashboard
        security_template = DashboardTemplate(
            name="Security & Compliance",
            description="Security metrics, access patterns, and compliance monitoring",
            tags=["security", "compliance", "audit"],
            refresh_interval="60s",
            time_range_from="now-7d",
        )

        security_template.add_visualization(
            VisualizationConfig(
                title="Authentication Failures",
                type=VisualizationType.TIME_SERIES,
                query="sum(rate(auth_failures_total[5m]))",
                description="Failed authentication attempts over time",
                unit="failures/sec",
                thresholds=[
                    Threshold(0.0, ThresholdLevel.GREEN),
                    Threshold(0.1, ThresholdLevel.YELLOW),
                    Threshold(1.0, ThresholdLevel.RED),
                ],
                width=12,
                height=8,
            )
        )

        security_template.add_visualization(
            VisualizationConfig(
                title="API Rate Limit Violations",
                type=VisualizationType.TABLE,
                query="topk(10, sum by (client_ip) (rate(rate_limit_exceeded_total[1h])))",
                description="Top IPs triggering rate limits",
                width=12,
                height=8,
                y_pos=8,
            )
        )

        self.templates["security"] = security_template

    def generate_dashboard(
        self,
        template_name: str,
        output_format: str = "grafana",
        customizations: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Generate a professional dashboard from template.

        Args:
            template_name: Name of the dashboard template to use
            output_format: Output format (grafana, datadog, newrelic)
            customizations: Optional customizations to apply

        Returns:
            Dashboard configuration in requested format
        """
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")

        template = self.templates[template_name]

        # Apply customizations
        if customizations:
            template = self._apply_customizations(template, customizations)

        # Generate based on output format
        if output_format == "grafana":
            return self._generate_grafana_dashboard(template)
        elif output_format == "datadog":
            return self._generate_datadog_dashboard(template)
        elif output_format == "newrelic":
            return self._generate_newrelic_dashboard(template)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    def _apply_customizations(
        self, template: DashboardTemplate, customizations: Dict[str, Any]
    ) -> DashboardTemplate:
        """Apply customizations to a dashboard template."""
        # Create a copy to avoid modifying the original
        import copy

        customized = copy.deepcopy(template)

        # Apply theme customizations
        if "theme" in customizations:
            for key, value in customizations["theme"].items():
                setattr(customized.theme, key, value)

        # Apply time range customizations
        if "time_range" in customizations:
            time_config = customizations["time_range"]
            customized.time_range_from = time_config.get(
                "from", customized.time_range_from
            )
            customized.time_range_to = time_config.get("to", customized.time_range_to)

        # Apply visualization customizations
        if "visualizations" in customizations:
            for viz_name, viz_config in customizations["visualizations"].items():
                for viz in customized.visualizations:
                    if viz.title == viz_name:
                        for key, value in viz_config.items():
                            setattr(viz, key, value)

        return customized

    def _generate_grafana_dashboard(
        self, template: DashboardTemplate
    ) -> Dict[str, Any]:
        """Generate Grafana-compatible dashboard JSON."""

        # Create panels from visualizations
        panels = []
        for i, viz in enumerate(template.visualizations):
            panel = {
                "id": i + 1,
                "title": viz.title,
                "type": self._map_visualization_type_grafana(viz.type),
                "gridPos": {
                    "h": viz.height,
                    "w": viz.width,
                    "x": viz.x_pos,
                    "y": viz.y_pos,
                },
                "targets": [
                    {"expr": viz.query, "legendFormat": "{{instance}}", "refId": "A"}
                ],
                "fieldConfig": {
                    "defaults": {
                        "unit": viz.unit,
                        "color": {"mode": "palette-classic"},
                        "thresholds": {
                            "steps": [
                                {
                                    "color": threshold.level.value,
                                    "value": threshold.value,
                                }
                                for threshold in viz.thresholds
                            ]
                        },
                    }
                },
                "options": {
                    "legend": {
                        "displayMode": "visible" if viz.legend_visible else "hidden"
                    },
                    "tooltip": {"mode": "shared" if viz.tooltip_shared else "single"},
                },
            }

            if viz.description:
                panel["description"] = viz.description

            panels.append(panel)

        # Create template variables
        templating = {
            "list": [
                {
                    "name": name,
                    "type": var_config.get("type", "query"),
                    "query": var_config["query"],
                    "label": var_config.get("label", name),
                    "multi": var_config.get("multi", False),
                    "includeAll": var_config.get("includeAll", False),
                    "refresh": var_config.get("refresh", "time"),
                    "datasource": {"type": "prometheus", "uid": "prometheus"},
                }
                for name, var_config in template.variables.items()
            ]
        }

        return {
            "dashboard": {
                "id": None,
                "title": template.name,
                "description": template.description,
                "tags": template.tags,
                "timezone": "browser",
                "panels": panels,
                "time": {
                    "from": template.time_range_from,
                    "to": template.time_range_to,
                },
                "timepicker": {
                    "refresh_intervals": [
                        "5s",
                        "10s",
                        "30s",
                        "1m",
                        "5m",
                        "15m",
                        "30m",
                        "1h",
                        "2h",
                        "1d",
                    ]
                },
                "refresh": template.refresh_interval,
                "templating": templating,
                "version": 1,
                "editable": True,
                "gnetId": None,
                "graphTooltip": 1,
                "links": [],
                "annotations": {"list": []},
                "schemaVersion": 30,
                "style": "dark",
                "uid": f"auto-{template.name.lower().replace(' ', '-')}",
            }
        }

    def _map_visualization_type_grafana(self, viz_type: VisualizationType) -> str:
        """Map visualization types to Grafana panel types."""
        mapping = {
            VisualizationType.SINGLE_STAT: "stat",
            VisualizationType.TIME_SERIES: "timeseries",
            VisualizationType.HEAT_MAP: "heatmap",
            VisualizationType.TABLE: "table",
            VisualizationType.PIE_CHART: "piechart",
            VisualizationType.BAR_GAUGE: "bargauge",
            VisualizationType.GAUGE: "gauge",
            VisualizationType.LOGS: "logs",
            VisualizationType.ALERT_LIST: "alertlist",
            VisualizationType.TEXT: "text",
            VisualizationType.NODE_GRAPH: "nodeGraph",
        }
        return mapping.get(viz_type, "timeseries")

    def _generate_datadog_dashboard(
        self, template: DashboardTemplate
    ) -> Dict[str, Any]:
        """Generate DataDog-compatible dashboard JSON."""
        # DataDog dashboard structure
        widgets = []

        for viz in template.visualizations:
            widget = {
                "definition": {
                    "type": self._map_visualization_type_datadog(viz.type),
                    "requests": [{"q": viz.query, "display_type": "line"}],
                    "title": viz.title,
                    "yAxis": {"scale": "linear"},
                },
                "layout": {
                    "x": viz.x_pos,
                    "y": viz.y_pos,
                    "width": viz.width,
                    "height": viz.height,
                },
            }
            widgets.append(widget)

        return {
            "title": template.name,
            "description": template.description,
            "widgets": widgets,
            "layout_type": "free",
            "is_read_only": False,
            "notify_list": [],
            "tags": template.tags,
        }

    def _map_visualization_type_datadog(self, viz_type: VisualizationType) -> str:
        """Map visualization types to DataDog widget types."""
        mapping = {
            VisualizationType.SINGLE_STAT: "query_value",
            VisualizationType.TIME_SERIES: "timeseries",
            VisualizationType.HEAT_MAP: "heatmap",
            VisualizationType.TABLE: "query_table",
            VisualizationType.PIE_CHART: "sunburst",
            VisualizationType.BAR_GAUGE: "query_value",
            VisualizationType.GAUGE: "query_value",
            VisualizationType.LOGS: "log_stream",
            VisualizationType.TEXT: "note",
        }
        return mapping.get(viz_type, "timeseries")

    def _generate_newrelic_dashboard(
        self, template: DashboardTemplate
    ) -> Dict[str, Any]:
        """Generate New Relic-compatible dashboard JSON."""
        pages = [
            {"name": template.name, "description": template.description, "widgets": []}
        ]

        for viz in template.visualizations:
            widget = {
                "title": viz.title,
                "layout": {
                    "column": viz.x_pos // 3 + 1,  # New Relic uses 12-column grid
                    "row": viz.y_pos // 3 + 1,
                    "width": viz.width // 3,
                    "height": viz.height // 3,
                },
                "visualization": {
                    "id": self._map_visualization_type_newrelic(viz.type)
                },
                "rawConfiguration": {
                    "nrqlQueries": [{"query": viz.query}],
                    "yAxisLeft": {"zero": True},
                },
            }
            pages[0]["widgets"].append(widget)

        return {
            "name": template.name,
            "description": template.description,
            "pages": pages,
            "permissions": "PUBLIC_READ_WRITE",
        }

    def _map_visualization_type_newrelic(self, viz_type: VisualizationType) -> str:
        """Map visualization types to New Relic visualization IDs."""
        mapping = {
            VisualizationType.SINGLE_STAT: "viz.billboard",
            VisualizationType.TIME_SERIES: "viz.line",
            VisualizationType.HEAT_MAP: "viz.heatmap",
            VisualizationType.TABLE: "viz.table",
            VisualizationType.PIE_CHART: "viz.pie",
            VisualizationType.BAR_GAUGE: "viz.bar",
            VisualizationType.GAUGE: "viz.billboard",
        }
        return mapping.get(viz_type, "viz.line")

    def export_dashboard_files(
        self,
        output_dir: Union[str, Path],
        template_names: List[str] | None = None,
        formats: List[str] | None = None,
    ) -> Dict[str, str]:
        """Export dashboard configurations to files.

        Args:
            output_dir: Directory to save dashboard files
            template_names: List of template names to export (default: all)
            formats: List of formats to export (default: ["grafana"])

        Returns:
            Dictionary mapping file paths to their contents
        """
        if formats is None:
            formats = ["grafana"]

        if template_names is None:
            template_names = list(self.templates.keys())

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        exported_files = {}

        for template_name in template_names:
            for format_name in formats:
                try:
                    dashboard_config = self.generate_dashboard(
                        template_name, format_name
                    )

                    filename = f"{template_name}-{format_name}.json"
                    file_path = output_path / filename

                    with open(file_path, "w") as f:
                        json.dump(dashboard_config, f, indent=2)

                    exported_files[str(file_path)] = json.dumps(
                        dashboard_config, indent=2
                    )
                    logger.info(
                        f"✅ Exported {template_name} dashboard ({format_name}) to {file_path}"
                    )

                except Exception as e:
                    logger.exception(
                        f"❌ Failed to export {template_name} ({format_name}): {e}"
                    )

        return exported_files

    def get_available_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available dashboard templates."""
        return {
            name: {
                "name": template.name,
                "description": template.description,
                "category": template.category,
                "tags": template.tags,
                "visualizations_count": len(template.visualizations),
                "variables_count": len(template.variables),
            }
            for name, template in self.templates.items()
        }

    def create_custom_template(
        self, name: str, description: str, visualizations: List[VisualizationConfig]
    ) -> DashboardTemplate:
        """Create a custom dashboard template."""
        template = DashboardTemplate(
            name=name, description=description, theme=self.theme
        )

        for viz in visualizations:
            template.add_visualization(viz)

        self.templates[name.lower().replace(" ", "_")] = template
        return template


# Portfolio showcase: AI-powered dashboard recommendations
class IntelligentDashboardRecommender:
    """AI-powered dashboard recommendations based on system usage patterns."""

    def __init__(self, generator: ProfessionalDashboardGenerator):
        self.generator = generator
        self.usage_patterns: Dict[str, Any] = {}

    def analyze_usage_patterns(self, metrics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze metrics to recommend optimal dashboard configurations."""
        patterns = {
            "primary_use_cases": [],
            "performance_bottlenecks": [],
            "cost_optimization_opportunities": [],
            "recommended_dashboards": [],
        }

        # Analyze request patterns
        if "request_rate" in metrics_data:
            if metrics_data["request_rate"] > 100:
                patterns["primary_use_cases"].append("high_traffic")
                patterns["recommended_dashboards"].append("technical")

        # Analyze AI usage
        if "ai_operations" in metrics_data:
            if metrics_data["ai_operations"]["cost_per_day"] > 10:
                patterns["primary_use_cases"].append("cost_sensitive")
                patterns["recommended_dashboards"].append("ai_operations")

        # Analyze error patterns
        if "error_rate" in metrics_data and metrics_data["error_rate"] > 0.01:
            patterns["performance_bottlenecks"].append("high_error_rate")
            patterns["recommended_dashboards"].append("security")

        return patterns

    def recommend_customizations(
        self, template_name: str, usage_patterns: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recommend template customizations based on usage patterns."""
        customizations = {}

        # High-traffic customizations
        if "high_traffic" in usage_patterns.get("primary_use_cases", []):
            customizations["visualizations"] = {
                "Response Time Percentiles": {
                    "refresh_interval": "5s",
                    "thresholds": [
                        Threshold(0.1, ThresholdLevel.GREEN),
                        Threshold(0.5, ThresholdLevel.YELLOW),
                        Threshold(1.0, ThresholdLevel.RED),
                    ],
                }
            }

        # Cost-sensitive customizations
        if "cost_sensitive" in usage_patterns.get("primary_use_cases", []):
            customizations["time_range"] = {
                "from": "now-7d",  # Longer time range for cost trending
                "to": "now",
            }

        return customizations
