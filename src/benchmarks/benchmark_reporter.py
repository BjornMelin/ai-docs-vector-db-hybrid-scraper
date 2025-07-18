"""Benchmark reporting and visualization tools.

This module provides comprehensive reporting capabilities for benchmark results
including HTML reports, charts, and performance analysis summaries.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


class BenchmarkReporter:
    """Advanced benchmark results reporter and visualizer."""

    def __init__(self):
        """Initialize benchmark reporter."""

    async def generate_html_report(self, results: Any) -> str:
        """Generate comprehensive HTML report from benchmark results.

        Args:
            results: BenchmarkResults object

        Returns:
            HTML report string

        """
        return self._build_html_template(results)

    def _build_html_template(self, results: Any) -> str:
        """Build HTML template with benchmark results."""
        # Extract key metrics for display
        latency_data = self._format_latency_metrics(results.latency_metrics)
        throughput_data = self._format_throughput_metrics(results.throughput_metrics)
        resource_data = self._format_resource_metrics(results.resource_metrics)
        accuracy_data = self._format_accuracy_metrics(results.accuracy_metrics)

        # Generate component results table
        component_table = self._generate_component_table(results.component_results)

        # Generate load test results
        load_test_section = self._generate_load_test_section(results.load_test_results)

        # Generate recommendations section
        recommendations_section = self._generate_recommendations_section(
            results.optimization_recommendations, results.performance_bottlenecks
        )

        # Build complete HTML
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced Hybrid Search Benchmark Report</title>
    <style>
        {self._get_css_styles()}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        <header class="report-header">
            <h1>🔍 Advanced Hybrid Search Benchmark Report</h1>
            <div class="report-metadata">
                <div class="metadata-item">
                    <strong>Benchmark:</strong> {results.benchmark_name}
                </div>
                <div class="metadata-item">
                    <strong>Duration:</strong> {results.duration_seconds:.2f} seconds
                </div>
                <div class="metadata-item">
                    <strong>Timestamp:</strong> {results.timestamp.strftime("%Y-%m-%d %H:%M:%S")}
                </div>
                <div class="metadata-item status-{"pass" if results.meets_targets else "fail"}">
                    <strong>Status:</strong> {"✅ PASS" if results.meets_targets else "❌ FAIL"}
                </div>
            </div>
        </header>

        <div class="summary-grid">
            <div class="summary-card">
                <h3>📊 Performance Summary</h3>
                {latency_data}
                {throughput_data}
            </div>

            <div class="summary-card">
                <h3>💾 Resource Usage</h3>
                {resource_data}
            </div>

            <div class="summary-card">
                <h3>🎯 ML Accuracy</h3>
                {accuracy_data}
            </div>
        </div>

        <section class="results-section">
            <h2>🔧 Component Performance</h2>
            {component_table}
        </section>

        <section class="results-section">
            <h2>⚡ Load Testing Results</h2>
            {load_test_section}
        </section>

        <section class="results-section">
            <h2>💡 Optimization Recommendations</h2>
            {recommendations_section}
        </section>

        <section class="results-section">
            <h2>📈 Performance Charts</h2>
            <div class="charts-grid">
                <div class="chart-container">
                    <canvas id="latencyChart"></canvas>
                </div>
                <div class="chart-container">
                    <canvas id="throughputChart"></canvas>
                </div>
            </div>
        </section>

        <footer class="report-footer">
            <p>Generated by Advanced Hybrid Search Benchmarking System</p>
            <p>Report generated at {datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S")}</p>
        </footer>
    </div>

    <script>
        {self._generate_charts_javascript(results)}
    </script>
</body>
</html>"""

    def _get_css_styles(self) -> str:
        """Get CSS styles for HTML report."""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8fafc;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        .report-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        .report-header h1 {
            font-size: 2.5rem;
            margin-bottom: 20px;
            text-align: center;
        }

        .report-metadata {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }

        .metadata-item {
            background: rgba(255,255,255,0.1);
            padding: 10px 15px;
            border-radius: 5px;
            backdrop-filter: blur(10px);
        }

        .status-pass {
            background: rgba(34, 197, 94, 0.2) !important;
        }

        .status-fail {
            background: rgba(239, 68, 68, 0.2) !important;
        }

        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .summary-card {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
        }

        .summary-card h3 {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.3rem;
        }

        .metric-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid #e5e7eb;
        }

        .metric-item:last-child {
            border-bottom: none;
        }

        .metric-value {
            font-weight: 600;
            color: #1f2937;
        }

        .metric-good {
            color: #059669;
        }

        .metric-warning {
            color: #d97706;
        }

        .metric-error {
            color: #dc2626;
        }

        .results-section {
            background: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .results-section h2 {
            color: #374151;
            margin-bottom: 20px;
            font-size: 1.8rem;
            border-bottom: 2px solid #e5e7eb;
            padding-bottom: 10px;
        }

        .component-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }

        .component-table th,
        .component-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e5e7eb;
        }

        .component-table th {
            background-color: #f9fafb;
            font-weight: 600;
            color: #374151;
        }

        .component-table tr:hover {
            background-color: #f9fafb;
        }

        .load-test-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }

        .load-test-card {
            background: #f9fafb;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #e5e7eb;
        }

        .load-test-card h4 {
            color: #667eea;
            margin-bottom: 10px;
        }

        .recommendations-list {
            list-style: none;
            padding: 0;
        }

        .recommendation-item {
            background: #f0f9ff;
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 8px;
            border-left: 4px solid #0ea5e9;
            display: flex;
            align-items: flex-start;
        }

        .recommendation-item::before {
            content: "💡";
            margin-right: 10px;
            font-size: 1.2em;
        }

        .bottleneck-item {
            background: #fef2f2;
            border-left-color: #ef4444;
        }

        .bottleneck-item::before {
            content: "⚠️";
        }

        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 30px;
            margin-top: 20px;
        }

        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            height: 400px;
        }

        .report-footer {
            text-align: center;
            color: #6b7280;
            margin-top: 40px;
            padding: 20px;
            border-top: 1px solid #e5e7eb;
        }

        @media (max-width: 768px) {
            .summary-grid {
                grid-template-columns: 1fr;
            }

            .charts-grid {
                grid-template-columns: 1fr;
            }

            .report-metadata {
                grid-template-columns: 1fr;
            }
        }
        """

    def _format_latency_metrics(self, latency_metrics: dict[str, float]) -> str:
        """Format latency metrics for display."""
        if not latency_metrics:
            return (
                "<div class='metric-item'><span>No latency data available</span></div>"
            )

        # Latency thresholds for UI styling
        latency_good_threshold = 100  # ms
        latency_warning_threshold = 300  # ms

        items = []
        for metric, value in latency_metrics.items():
            css_class = (
                "metric-good"
                if value < latency_good_threshold
                else "metric-warning"
                if value < latency_warning_threshold
                else "metric-error"
            )
            items.append(f"""
                <div class="metric-item">
                    <span>{metric.replace("_", " ").title()}</span>
                    <span class="metric-value {css_class}">{value:.1f}ms</span>
                </div>
            """)

        return "".join(items)

    def _format_throughput_metrics(self, throughput_metrics: dict[str, float]) -> str:
        """Format throughput metrics for display."""
        if not throughput_metrics:
            return "<div class='metric-item'><span>No throughput data available</span></div>"

        items = []
        for metric, value in throughput_metrics.items():
            css_class = (
                "metric-good"
                if value > 100
                else "metric-warning"
                if value > 50
                else "metric-error"
            )
            items.append(f"""
                <div class="metric-item">
                    <span>{metric.replace("_", " ").title()}</span>
                    <span class="metric-value {css_class}">{value:.1f} QPS</span>
                </div>
            """)

        return "".join(items)

    def _format_resource_metrics(self, resource_metrics: dict[str, float]) -> str:
        """Format resource metrics for display."""
        if not resource_metrics:
            return (
                "<div class='metric-item'><span>No resource data available</span></div>"
            )

        items = []
        for metric, value in resource_metrics.items():
            if "memory" in metric.lower():
                css_class = (
                    "metric-good"
                    if value < 1000
                    else "metric-warning"
                    if value < 2000
                    else "metric-error"
                )
                unit = "MB"
            elif "cpu" in metric.lower():
                css_class = (
                    "metric-good"
                    if value < 50
                    else "metric-warning"
                    if value < 80
                    else "metric-error"
                )
                unit = "%"
            else:
                css_class = "metric-value"
                unit = ""

            items.append(f"""
                <div class="metric-item">
                    <span>{metric.replace("_", " ").title()}</span>
                    <span class="metric-value {css_class}">{value:.1f}{unit}</span>
                </div>
            """)

        return "".join(items)

    def _format_accuracy_metrics(self, accuracy_metrics: dict[str, float]) -> str:
        """Format accuracy metrics for display."""
        if not accuracy_metrics:
            return (
                "<div class='metric-item'><span>No accuracy data available</span></div>"
            )

        items = []
        for metric, value in accuracy_metrics.items():
            percentage = value * 100
            css_class = (
                "metric-good"
                if percentage > 80
                else "metric-warning"
                if percentage > 60
                else "metric-error"
            )
            items.append(f"""
                <div class="metric-item">
                    <span>{metric.replace("_", " ").title()}</span>
                    <span class="metric-value {css_class}">{percentage:.1f}%</span>
                </div>
            """)

        return "".join(items)

    def _generate_component_table(self, component_results: dict[str, Any]) -> str:
        """Generate component performance table."""
        if not component_results:
            return "<p>No component benchmark data available.</p>"

        rows = []
        for component_name, result in component_results.items():
            # Handle different result formats
            if hasattr(result, "model_dump"):
                result_dict = result.model_dump()
            elif isinstance(result, dict):
                result_dict = result
            else:
                continue

            avg_latency = result_dict.get("avg_latency_ms", 0)
            p95_latency = result_dict.get("p95_latency_ms", 0)
            throughput = result_dict.get("throughput_per_second", 0)
            error_rate = result_dict.get("error_rate", 0) * 100

            latency_class = (
                "metric-good"
                if avg_latency < 50
                else "metric-warning"
                if avg_latency < 150
                else "metric-error"
            )
            throughput_class = (
                "metric-good"
                if throughput > 10
                else "metric-warning"
                if throughput > 5
                else "metric-error"
            )
            error_class = (
                "metric-good"
                if error_rate < 1
                else "metric-warning"
                if error_rate < 5
                else "metric-error"
            )

            rows.append(f"""
                <tr>
                    <td><strong>{component_name.replace("_", " ").title()}</strong></td>
                    <td class="{latency_class}">{avg_latency:.1f}ms</td>
                    <td class="{latency_class}">{p95_latency:.1f}ms</td>
                    <td class="{throughput_class}">{throughput:.1f}</td>
                    <td class="{error_class}">{error_rate:.1f}%</td>
                </tr>
            """)

        return f"""
        <table class="component-table">
            <thead>
                <tr>
                    <th>Component</th>
                    <th>Avg Latency</th>
                    <th>P95 Latency</th>
                    <th>Throughput</th>
                    <th>Error Rate</th>
                </tr>
            </thead>
            <tbody>
                {"".join(rows)}
            </tbody>
        </table>
        """

    def _generate_load_test_section(self, load_test_results: dict[str, Any]) -> str:
        """Generate load test results section."""
        if not load_test_results:
            return "<p>No load test data available.</p>"

        cards = []
        for test_name, result in load_test_results.items():
            if hasattr(result, "model_dump"):
                result_dict = result.model_dump()
            elif isinstance(result, dict):
                result_dict = result
            else:
                continue

            total_requests = result_dict.get("total_requests", 0)
            successful_requests = result_dict.get("successful_requests", 0)
            avg_response_time = result_dict.get("avg_response_time_ms", 0)
            requests_per_second = result_dict.get("requests_per_second", 0)

            cards.append(f"""
                <div class="load-test-card">
                    <h4>{test_name.replace("_", " ").title()}</h4>
                    <div class="metric-item">
                        <span>Total Requests:</span>
                        <span class="metric-value">{total_requests:,}</span>
                    </div>
                    <div class="metric-item">
                        <span>Success Rate:</span>
                        <span class="metric-value metric-good">{((successful_requests / max(total_requests, 1)) * 100):.1f}%</span>
                    </div>
                    <div class="metric-item">
                        <span>Avg Response Time:</span>
                        <span class="metric-value">{avg_response_time:.1f}ms</span>
                    </div>
                    <div class="metric-item">
                        <span>Throughput:</span>
                        <span class="metric-value">{requests_per_second:.1f} QPS</span>
                    </div>
                </div>
            """)

        return f'<div class="load-test-grid">{"".join(cards)}</div>'

    def _generate_recommendations_section(
        self, recommendations: list[str], bottlenecks: list[str]
    ) -> str:
        """Generate recommendations and bottlenecks section."""
        content = []

        if bottlenecks:
            content.append("<h3>⚠️ Performance Bottlenecks</h3>")
            content.append('<ul class="recommendations-list">')
            content.extend(
                [
                    f'<li class="recommendation-item bottleneck-item">{bottleneck}</li>'
                    for bottleneck in bottlenecks
                ]
            )
            content.append("</ul>")

        if recommendations:
            content.append("<h3>💡 Optimization Recommendations</h3>")
            content.append('<ul class="recommendations-list">')
            content.extend(
                [
                    f'<li class="recommendation-item">{recommendation}</li>'
                    for recommendation in recommendations
                ]
            )
            content.append("</ul>")

        if not content:
            content.append(
                "<p>No specific recommendations available. System performance appears to be within acceptable ranges.</p>"
            )

        return "".join(content)

    def _generate_charts_javascript(self, results: Any) -> str:
        """Generate JavaScript for performance charts."""
        # Extract data for charts
        component_names = (
            list(results.component_results.keys()) if results.component_results else []
        )
        latencies = []
        throughputs = []

        for component_name in component_names:
            result = results.component_results[component_name]
            if hasattr(result, "model_dump"):
                result_dict = result.model_dump()
            elif isinstance(result, dict):
                result_dict = result
            else:
                result_dict = {}

            latencies.append(result_dict.get("avg_latency_ms", 0))
            throughputs.append(result_dict.get("throughput_per_second", 0))

        return f"""
        // Latency Chart
        const latencyCtx = document.getElementById('latencyChart').getContext('2d');
        new Chart(latencyCtx, {{
            type: 'bar',
            data: {{
                labels: {json.dumps([name.replace("_", " ").title() for name in component_names])},
                datasets: [{{
                    label: 'Average Latency (ms)',
                    data: {json.dumps(latencies)},
                    backgroundColor: 'rgba(102, 126, 234, 0.8)',
                    borderColor: 'rgba(102, 126, 234, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Component Latency Comparison'
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Latency (ms)'
                        }}
                    }}
                }}
            }}
        }});

        // Throughput Chart
        const throughputCtx = document.getElementById('throughputChart').getContext('2d');
        new Chart(throughputCtx, {{
            type: 'bar',
            data: {{
                labels: {json.dumps([name.replace("_", " ").title() for name in component_names])},
                datasets: [{{
                    label: 'Throughput (ops/sec)',
                    data: {json.dumps(throughputs)},
                    backgroundColor: 'rgba(34, 197, 94, 0.8)',
                    borderColor: 'rgba(34, 197, 94, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Component Throughput Comparison'
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Operations per Second'
                        }}
                    }}
                }}
            }}
        }});
        """

    def generate_json_report(self, results: Any) -> str:
        """Generate JSON report from benchmark results.

        Args:
            results: BenchmarkResults object

        Returns:
            JSON report string

        """
        if hasattr(results, "model_dump"):
            report_data = results.model_dump()
        else:
            report_data = results

        # Add metadata
        report_data["report_metadata"] = {
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "report_version": "1.0.0",
            "format": "json",
        }

        return json.dumps(report_data, indent=2, default=str)

    def generate_csv_summary(self, results: Any) -> str:
        """Generate CSV summary of key metrics.

        Args:
            results: BenchmarkResults object

        Returns:
            CSV summary string

        """
        csv_lines = ["Metric,Value,Unit,Status"]

        # Add latency metrics
        for metric, value in results.latency_metrics.items():
            status = "PASS" if value < 300 else "FAIL"
            csv_lines.append(f"{metric},{value:.2f},ms,{status}")

        # Add throughput metrics
        for metric, value in results.throughput_metrics.items():
            status = "PASS" if value > 50 else "FAIL"
            csv_lines.append(f"{metric},{value:.2f},QPS,{status}")

        # Add resource metrics
        for metric, value in results.resource_metrics.items():
            unit = "MB" if "memory" in metric.lower() else "%"
            status = "PASS" if value < 2000 else "FAIL"
            csv_lines.append(f"{metric},{value:.2f},{unit},{status}")

        # Add accuracy metrics
        for metric, value in results.accuracy_metrics.items():
            status = "PASS" if value > 0.8 else "FAIL"
            csv_lines.append(f"{metric},{value:.3f},ratio,{status}")

        return "\n".join(csv_lines)

    async def save_reports(
        self, results: Any, output_dir: Path, formats: list[str] | None = None
    ) -> dict[str, str]:
        """Save benchmark reports in multiple formats.

        Args:
            results: BenchmarkResults object
            output_dir: Output directory
            formats: List of formats to generate

        Returns:
            Dictionary mapping format to file path

        """
        if formats is None:
            formats = ["html", "json"]

        output_dir.mkdir(parents=True, exist_ok=True)
        saved_files = {}

        timestamp = int(datetime.now(tz=UTC).timestamp())

        if "html" in formats:
            html_report = await self.generate_html_report(results)
            html_file = output_dir / f"benchmark_report_{timestamp}.html"
            with html_file.open("w", encoding="utf-8") as f:
                f.write(html_report)
            saved_files["html"] = str(html_file)

        if "json" in formats:
            json_report = self.generate_json_report(results)
            json_file = output_dir / f"benchmark_results_{timestamp}.json"
            with json_file.open("w", encoding="utf-8") as f:
                f.write(json_report)
            saved_files["json"] = str(json_file)

        if "csv" in formats:
            csv_report = self.generate_csv_summary(results)
            csv_file = output_dir / f"benchmark_summary_{timestamp}.csv"
            with csv_file.open("w", encoding="utf-8") as f:
                f.write(csv_report)
            saved_files["csv"] = str(csv_file)

        logger.info("Saved benchmark reports: %s", list(saved_files.keys()))
        return saved_files
