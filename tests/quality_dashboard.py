"""Quality Engineering Dashboard and Metrics Collection.

This module provides comprehensive quality metrics collection, analysis, and reporting
for the Quality Engineering Center of Excellence. It includes advanced metrics
calculation, trend analysis, and quality gate enforcement.
"""

import json
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pydantic import BaseModel


@dataclass
class QualityMetrics:
    """Comprehensive quality metrics data structure."""

    timestamp: datetime = field(default_factory=datetime.now)

    # Test Execution Metrics
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    execution_time_seconds: float = 0.0

    # Coverage Metrics
    line_coverage_percent: float = 0.0
    branch_coverage_percent: float = 0.0
    function_coverage_percent: float = 0.0

    # Code Quality Metrics
    cyclomatic_complexity: float = 0.0
    maintainability_index: float = 0.0
    technical_debt_minutes: int = 0

    # Security Metrics
    security_vulnerabilities: int = 0
    security_score: float = 100.0

    # Performance Metrics
    avg_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0
    throughput_rps: float = 0.0

    # Contract Testing Metrics
    contract_tests: int = 0
    contract_violations: int = 0
    contract_compliance_percent: float = 100.0

    # Chaos Engineering Metrics
    chaos_experiments: int = 0
    resilience_score: float = 100.0
    mttr_minutes: float = 0.0

    # Property-Based Testing Metrics
    property_tests: int = 0
    property_violations: int = 0
    hypothesis_examples: int = 0
    shrinking_attempts: int = 0

    # Mutation Testing Metrics
    mutation_score: float = 0.0
    mutants_killed: int = 0
    mutants_survived: int = 0

    @property
    def test_success_rate(self) -> float:
        """Calculate test success rate percentage."""
        if self.total_tests == 0:
            return 100.0
        return (self.passed_tests / self.total_tests) * 100

    @property
    def overall_quality_score(self) -> float:
        """Calculate overall quality score (0-100)."""
        weights = {
            "test_success_rate": 0.25,
            "line_coverage_percent": 0.20,
            "security_score": 0.20,
            "contract_compliance_percent": 0.15,
            "resilience_score": 0.10,
            "mutation_score": 0.10,
        }

        score = (
            self.test_success_rate * weights["test_success_rate"]
            + self.line_coverage_percent * weights["line_coverage_percent"]
            + self.security_score * weights["security_score"]
            + self.contract_compliance_percent * weights["contract_compliance_percent"]
            + self.resilience_score * weights["resilience_score"]
            + self.mutation_score * weights["mutation_score"]
        )

        return min(100.0, max(0.0, score))


class QualityTrend(BaseModel):
    """Quality trend analysis data."""

    metric_name: str
    current_value: float
    previous_value: float
    trend_direction: str  # 'up', 'down', 'stable'
    trend_percentage: float
    is_improving: bool


class QualityGate(BaseModel):
    """Quality gate configuration and validation."""

    name: str
    metric: str
    threshold: float
    operator: str  # 'gt', 'lt', 'gte', 'lte', 'eq'
    is_blocking: bool = True
    description: str = ""


class QualityDashboard:
    """Advanced Quality Engineering Dashboard."""

    def __init__(self, db_path: Path | None = None):
        """Initialize quality dashboard with database storage."""
        self.db_path = db_path or Path("tests/fixtures/data/quality_metrics.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

        # Quality gates configuration
        self.quality_gates = [
            QualityGate(
                name="Test Success Rate",
                metric="test_success_rate",
                threshold=95.0,
                operator="gte",
                description="Test success rate must be >= 95%",
            ),
            QualityGate(
                name="Code Coverage",
                metric="line_coverage_percent",
                threshold=90.0,
                operator="gte",
                description="Line coverage must be >= 90%",
            ),
            QualityGate(
                name="Security Score",
                metric="security_score",
                threshold=95.0,
                operator="gte",
                description="Security score must be >= 95%",
            ),
            QualityGate(
                name="Contract Compliance",
                metric="contract_compliance_percent",
                threshold=100.0,
                operator="eq",
                description="Contract compliance must be 100%",
            ),
            QualityGate(
                name="Mutation Score",
                metric="mutation_score",
                threshold=80.0,
                operator="gte",
                description="Mutation score must be >= 80%",
            ),
            QualityGate(
                name="Performance SLA",
                metric="p95_response_time_ms",
                threshold=100.0,
                operator="lte",
                description="95th percentile response time must be <= 100ms",
            ),
        ]

    def _init_database(self) -> None:
        """Initialize SQLite database for metrics storage."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quality_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    build_id TEXT,
                    commit_hash TEXT,
                    branch_name TEXT
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON quality_metrics(timestamp)
            """)

    def record_metrics(
        self,
        metrics: QualityMetrics,
        build_id: str | None = None,
        commit_hash: str | None = None,
        branch_name: str | None = None,
    ) -> None:
        """Record quality metrics to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO quality_metrics 
                (timestamp, metrics_json, build_id, commit_hash, branch_name)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    metrics.timestamp.isoformat(),
                    json.dumps(metrics.__dict__, default=str),
                    build_id,
                    commit_hash,
                    branch_name,
                ),
            )

    def get_latest_metrics(self) -> QualityMetrics | None:
        """Get the most recent quality metrics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT metrics_json FROM quality_metrics 
                ORDER BY timestamp DESC LIMIT 1
            """)
            row = cursor.fetchone()

            if row:
                data = json.loads(row[0])
                data["timestamp"] = datetime.fromisoformat(data["timestamp"])
                return QualityMetrics(**data)

        return None

    def get_historical_metrics(self, days: int = 30) -> List[QualityMetrics]:
        """Get historical metrics for trend analysis."""
        cutoff_date = datetime.now() - timedelta(days=days)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT metrics_json FROM quality_metrics 
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            """,
                (cutoff_date.isoformat(),),
            )

            metrics_list = []
            for row in cursor.fetchall():
                data = json.loads(row[0])
                data["timestamp"] = datetime.fromisoformat(data["timestamp"])
                metrics_list.append(QualityMetrics(**data))

            return metrics_list

    def calculate_trends(self, days: int = 7) -> List[QualityTrend]:
        """Calculate quality trends over specified period."""
        metrics_history = self.get_historical_metrics(days)

        if len(metrics_history) < 2:
            return []

        latest = metrics_history[0]
        baseline = metrics_history[-1]

        trends = []

        # Key metrics to track trends
        trend_metrics = {
            "test_success_rate": "Test Success Rate",
            "line_coverage_percent": "Line Coverage",
            "security_score": "Security Score",
            "contract_compliance_percent": "Contract Compliance",
            "mutation_score": "Mutation Score",
            "overall_quality_score": "Overall Quality Score",
        }

        for attr, display_name in trend_metrics.items():
            current_value = getattr(latest, attr)
            previous_value = getattr(baseline, attr)

            if previous_value == 0:
                trend_percentage = 0.0
                trend_direction = "stable"
            else:
                trend_percentage = (
                    (current_value - previous_value) / previous_value
                ) * 100

                if abs(trend_percentage) < 1.0:
                    trend_direction = "stable"
                elif trend_percentage > 0:
                    trend_direction = "up"
                else:
                    trend_direction = "down"

            # Determine if trend is improving (higher is better for most metrics)
            is_improving = trend_direction == "up"

            trends.append(
                QualityTrend(
                    metric_name=display_name,
                    current_value=current_value,
                    previous_value=previous_value,
                    trend_direction=trend_direction,
                    trend_percentage=trend_percentage,
                    is_improving=is_improving,
                )
            )

        return trends

    def validate_quality_gates(self, metrics: QualityMetrics) -> Dict[str, Any]:
        """Validate metrics against quality gates."""
        results = {"passed": True, "gates": [], "blocking_failures": [], "warnings": []}

        for gate in self.quality_gates:
            metric_value = getattr(metrics, gate.metric, None)

            if metric_value is None:
                continue

            passed = self._evaluate_gate(metric_value, gate.threshold, gate.operator)

            gate_result = {
                "name": gate.name,
                "metric": gate.metric,
                "value": metric_value,
                "threshold": gate.threshold,
                "operator": gate.operator,
                "passed": passed,
                "is_blocking": gate.is_blocking,
                "description": gate.description,
            }

            results["gates"].append(gate_result)

            if not passed:
                if gate.is_blocking:
                    results["blocking_failures"].append(gate_result)
                    results["passed"] = False
                else:
                    results["warnings"].append(gate_result)

        return results

    def _evaluate_gate(self, value: float, threshold: float, operator: str) -> bool:
        """Evaluate a single quality gate condition."""
        operators = {
            "gt": lambda v, t: v > t,
            "lt": lambda v, t: v < t,
            "gte": lambda v, t: v >= t,
            "lte": lambda v, t: v <= t,
            "eq": lambda v, t: abs(v - t) < 0.01,  # Float equality with tolerance
        }

        return operators.get(operator, lambda v, t: False)(value, threshold)

    def generate_quality_report(self, format: str = "json") -> str:
        """Generate comprehensive quality report."""
        latest_metrics = self.get_latest_metrics()

        if not latest_metrics:
            return "No metrics available"

        trends = self.calculate_trends()
        quality_gates = self.validate_quality_gates(latest_metrics)

        report_data = {
            "timestamp": latest_metrics.timestamp.isoformat(),
            "overall_quality_score": latest_metrics.overall_quality_score,
            "metrics": {
                "test_execution": {
                    "total_tests": latest_metrics.total_tests,
                    "success_rate": latest_metrics.test_success_rate,
                    "execution_time": latest_metrics.execution_time_seconds,
                },
                "coverage": {
                    "line_coverage": latest_metrics.line_coverage_percent,
                    "branch_coverage": latest_metrics.branch_coverage_percent,
                    "function_coverage": latest_metrics.function_coverage_percent,
                },
                "security": {
                    "score": latest_metrics.security_score,
                    "vulnerabilities": latest_metrics.security_vulnerabilities,
                },
                "performance": {
                    "avg_response_time": latest_metrics.avg_response_time_ms,
                    "p95_response_time": latest_metrics.p95_response_time_ms,
                    "throughput": latest_metrics.throughput_rps,
                },
                "contracts": {
                    "compliance_percent": latest_metrics.contract_compliance_percent,
                    "violations": latest_metrics.contract_violations,
                },
                "mutation_testing": {
                    "score": latest_metrics.mutation_score,
                    "mutants_killed": latest_metrics.mutants_killed,
                    "mutants_survived": latest_metrics.mutants_survived,
                },
            },
            "trends": [trend.dict() for trend in trends],
            "quality_gates": quality_gates,
            "recommendations": self._generate_recommendations(latest_metrics, trends),
        }

        if format.lower() == "json":
            return json.dumps(report_data, indent=2, default=str)
        elif format.lower() == "html":
            return self._generate_html_report(report_data)
        else:
            return str(report_data)

    def _generate_recommendations(
        self, metrics: QualityMetrics, trends: List[QualityTrend]
    ) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []

        # Test success rate recommendations
        if metrics.test_success_rate < 95:
            recommendations.append(
                f"Improve test success rate ({metrics.test_success_rate:.1f}%) by investigating failing tests"
            )

        # Coverage recommendations
        if metrics.line_coverage_percent < 90:
            recommendations.append(
                f"Increase code coverage ({metrics.line_coverage_percent:.1f}%) by adding tests for uncovered code"
            )

        # Performance recommendations
        if metrics.p95_response_time_ms > 100:
            recommendations.append(
                f"Optimize performance - 95th percentile response time is {metrics.p95_response_time_ms:.1f}ms"
            )

        # Security recommendations
        if metrics.security_vulnerabilities > 0:
            recommendations.append(
                f"Address {metrics.security_vulnerabilities} security vulnerabilities"
            )

        # Mutation testing recommendations
        if metrics.mutation_score < 80:
            recommendations.append(
                f"Improve test quality - mutation score is {metrics.mutation_score:.1f}%"
            )

        # Trend-based recommendations
        declining_trends = [
            t for t in trends if t.trend_direction == "down" and not t.is_improving
        ]
        for trend in declining_trends:
            recommendations.append(
                f"Address declining {trend.metric_name} (down {abs(trend.trend_percentage):.1f}%)"
            )

        if not recommendations:
            recommendations.append(
                "Quality metrics are excellent! Continue maintaining current standards."
            )

        return recommendations

    def _generate_html_report(self, data: Dict[str, Any]) -> str:
        """Generate HTML quality report."""
        overall_score = data["overall_quality_score"]
        score_class = (
            "excellent"
            if overall_score >= 90
            else "good"
            if overall_score >= 80
            else "warning"
        )

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Quality Engineering Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #2c3e50; color: white; padding: 20px; }}
                .score {{ font-size: 48px; font-weight: bold; }}
                .excellent {{ color: #27ae60; }}
                .good {{ color: #f39c12; }}
                .warning {{ color: #e74c3c; }}
                .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
                .metric-card {{ border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
                .trend-up {{ color: #27ae60; }}
                .trend-down {{ color: #e74c3c; }}
                .trend-stable {{ color: #7f8c8d; }}
                .quality-gate.passed {{ color: #27ae60; }}
                .quality-gate.failed {{ color: #e74c3c; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Quality Engineering Dashboard</h1>
                <div class="score {score_class}">Quality Score: {overall_score:.1f}%</div>
                <p>Generated: {data["timestamp"]}</p>
            </div>
            
            <div class="metric-grid">
        """

        # Add metric cards
        metrics = data["metrics"]
        for category, values in metrics.items():
            html += f"""
                <div class="metric-card">
                    <h3>{category.replace("_", " ").title()}</h3>
            """
            for key, value in values.items():
                html += (
                    f"<p><strong>{key.replace('_', ' ').title()}:</strong> {value}</p>"
                )
            html += "</div>"

        html += """
            </div>
            
            <h2>Quality Trends</h2>
            <ul>
        """

        # Add trends
        for trend in data["trends"]:
            trend_class = f"trend-{trend['trend_direction']}"
            html += f"""
                <li class="{trend_class}">
                    {trend["metric_name"]}: {trend["current_value"]:.1f} 
                    ({trend["trend_direction"]} {abs(trend["trend_percentage"]):.1f}%)
                </li>
            """

        html += """
            </ul>
            
            <h2>Quality Gates</h2>
            <ul>
        """

        # Add quality gates
        for gate in data["quality_gates"]["gates"]:
            gate_class = "passed" if gate["passed"] else "failed"
            status = "✓" if gate["passed"] else "✗"
            html += f"""
                <li class="quality-gate {gate_class}">
                    {status} {gate["name"]}: {gate["value"]} {gate["operator"]} {gate["threshold"]}
                </li>
            """

        html += """
            </ul>
            
            <h2>Recommendations</h2>
            <ul>
        """

        # Add recommendations
        for rec in data["recommendations"]:
            html += f"<li>{rec}</li>"

        html += """
            </ul>
        </body>
        </html>
        """

        return html

    def create_quality_visualizations(self, output_dir: Path) -> None:
        """Create quality metric visualizations."""
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_history = self.get_historical_metrics(30)

        if len(metrics_history) < 2:
            return

        # Prepare data for visualization
        df_data = []
        for m in metrics_history:
            df_data.append(
                {
                    "timestamp": m.timestamp,
                    "quality_score": m.overall_quality_score,
                    "test_success_rate": m.test_success_rate,
                    "coverage": m.line_coverage_percent,
                    "security_score": m.security_score,
                    "mutation_score": m.mutation_score,
                    "response_time": m.p95_response_time_ms,
                }
            )

        df = pd.DataFrame(df_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Set style
        plt.style.use("seaborn-v0_8")

        # Create multi-panel visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(
            "Quality Engineering Metrics Dashboard", fontsize=16, fontweight="bold"
        )

        # Quality Score Trend
        axes[0, 0].plot(df["timestamp"], df["quality_score"], "b-", linewidth=2)
        axes[0, 0].set_title("Overall Quality Score")
        axes[0, 0].set_ylabel("Score (%)")
        axes[0, 0].grid(True, alpha=0.3)

        # Test Success Rate
        axes[0, 1].plot(df["timestamp"], df["test_success_rate"], "g-", linewidth=2)
        axes[0, 1].set_title("Test Success Rate")
        axes[0, 1].set_ylabel("Success Rate (%)")
        axes[0, 1].grid(True, alpha=0.3)

        # Code Coverage
        axes[0, 2].plot(df["timestamp"], df["coverage"], "orange", linewidth=2)
        axes[0, 2].set_title("Code Coverage")
        axes[0, 2].set_ylabel("Coverage (%)")
        axes[0, 2].grid(True, alpha=0.3)

        # Security Score
        axes[1, 0].plot(df["timestamp"], df["security_score"], "r-", linewidth=2)
        axes[1, 0].set_title("Security Score")
        axes[1, 0].set_ylabel("Score (%)")
        axes[1, 0].grid(True, alpha=0.3)

        # Mutation Score
        axes[1, 1].plot(df["timestamp"], df["mutation_score"], "purple", linewidth=2)
        axes[1, 1].set_title("Mutation Testing Score")
        axes[1, 1].set_ylabel("Score (%)")
        axes[1, 1].grid(True, alpha=0.3)

        # Response Time
        axes[1, 2].plot(df["timestamp"], df["response_time"], "brown", linewidth=2)
        axes[1, 2].set_title("P95 Response Time")
        axes[1, 2].set_ylabel("Time (ms)")
        axes[1, 2].grid(True, alpha=0.3)

        # Format x-axis for all subplots
        for ax in axes.flat:
            ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(
            output_dir / "quality_metrics_dashboard.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # Create quality gate status chart
        latest_metrics = self.get_latest_metrics()
        if latest_metrics:
            gate_results = self.validate_quality_gates(latest_metrics)

            fig, ax = plt.subplots(figsize=(10, 6))

            gate_names = [gate["name"] for gate in gate_results["gates"]]
            gate_values = [gate["value"] for gate in gate_results["gates"]]
            gate_thresholds = [gate["threshold"] for gate in gate_results["gates"]]
            gate_passed = [gate["passed"] for gate in gate_results["gates"]]

            x = range(len(gate_names))

            # Create bars with different colors for passed/failed
            colors = ["green" if passed else "red" for passed in gate_passed]
            bars = ax.bar(
                x, gate_values, color=colors, alpha=0.7, label="Current Value"
            )

            # Add threshold line
            ax.plot(x, gate_thresholds, "ko-", label="Threshold")

            ax.set_xlabel("Quality Gates")
            ax.set_ylabel("Value")
            ax.set_title("Quality Gate Status")
            ax.set_xticks(x)
            ax.set_xticklabels(gate_names, rotation=45, ha="right")
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(
                output_dir / "quality_gates_status.png", dpi=300, bbox_inches="tight"
            )
            plt.close()


def create_sample_metrics() -> QualityMetrics:
    """Create sample quality metrics for testing."""
    return QualityMetrics(
        total_tests=1547,
        passed_tests=1521,
        failed_tests=14,
        skipped_tests=12,
        execution_time_seconds=234.5,
        line_coverage_percent=94.2,
        branch_coverage_percent=89.7,
        function_coverage_percent=96.1,
        cyclomatic_complexity=2.3,
        maintainability_index=87.4,
        technical_debt_minutes=45,
        security_vulnerabilities=0,
        security_score=98.5,
        avg_response_time_ms=67.3,
        p95_response_time_ms=89.2,
        p99_response_time_ms=145.7,
        throughput_rps=1250.0,
        contract_tests=234,
        contract_violations=0,
        contract_compliance_percent=100.0,
        chaos_experiments=12,
        resilience_score=96.8,
        mttr_minutes=3.2,
        property_tests=45,
        property_violations=0,
        hypothesis_examples=15234,
        shrinking_attempts=89,
        mutation_score=87.3,
        mutants_killed=523,
        mutants_survived=76,
    )


if __name__ == "__main__":
    # Demo usage
    dashboard = QualityDashboard()

    # Record sample metrics
    sample_metrics = create_sample_metrics()
    dashboard.record_metrics(
        sample_metrics,
        build_id="build-2025-001",
        commit_hash="abc123",
        branch_name="main",
    )

    # Generate reports
    json_report = dashboard.generate_quality_report("json")
    print("Quality Report (JSON):")
    print(json_report)

    # Create visualizations
    dashboard.create_quality_visualizations(Path("tests/fixtures/data"))

    print("\nQuality dashboard demonstration completed!")
