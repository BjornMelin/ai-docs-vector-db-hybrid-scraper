#!/usr/bin/env python3
"""Test performance monitoring dashboard and regression detection."""

import json
import sqlite3
import time
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


class TestPerformanceDatabase:
    """SQLite database for storing test performance metrics."""
    
    def __init__(self, db_path: str = "test_performance.db"):
        self.db_path = Path(db_path)
        self.init_database()
    
    def init_database(self):
        """Initialize performance tracking database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS test_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    git_commit TEXT,
                    branch TEXT,
                    total_tests INTEGER,
                    total_time REAL,
                    average_time REAL,
                    parallel_workers INTEGER,
                    success_rate REAL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS test_timings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    test_name TEXT,
                    duration REAL,
                    category TEXT,
                    FOREIGN KEY (run_id) REFERENCES test_runs (id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    alert_type TEXT,
                    test_name TEXT,
                    current_value REAL,
                    baseline_value REAL,
                    threshold_exceeded REAL,
                    message TEXT
                )
            """)
    
    def record_test_run(self, results: Dict) -> int:
        """Record a test run and return run ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO test_runs 
                (timestamp, git_commit, branch, total_tests, total_time, 
                 average_time, parallel_workers, success_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                time.time(),
                results.get("git_commit", "unknown"),
                results.get("branch", "unknown"),
                results["total_tests"],
                results["total_time"],
                results["average_time"],
                results.get("parallel_workers", 1),
                results.get("success_rate", 1.0),
            ))
            return cursor.lastrowid
    
    def record_test_timings(self, run_id: int, timings: List[Dict]):
        """Record individual test timings."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany("""
                INSERT INTO test_timings (run_id, test_name, duration, category)
                VALUES (?, ?, ?, ?)
            """, [
                (run_id, timing["test"], timing["duration"], timing.get("category", "unknown"))
                for timing in timings
            ])
    
    def record_alert(self, alert_type: str, test_name: str, current: float, 
                    baseline: float, threshold: float, message: str):
        """Record a performance alert."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO performance_alerts 
                (timestamp, alert_type, test_name, current_value, baseline_value, 
                 threshold_exceeded, message)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                time.time(), alert_type, test_name, current, baseline, threshold, message
            ))
    
    def get_recent_runs(self, days: int = 30) -> List[Dict]:
        """Get recent test runs."""
        cutoff = time.time() - (days * 24 * 3600)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM test_runs 
                WHERE timestamp > ? 
                ORDER BY timestamp DESC
            """, (cutoff,))
            
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_test_history(self, test_name: str, days: int = 30) -> List[Dict]:
        """Get performance history for a specific test."""
        cutoff = time.time() - (days * 24 * 3600)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT tr.timestamp, tr.git_commit, tt.duration
                FROM test_timings tt
                JOIN test_runs tr ON tt.run_id = tr.id
                WHERE tt.test_name = ? AND tr.timestamp > ?
                ORDER BY tr.timestamp DESC
            """, (test_name, cutoff))
            
            return [
                {"timestamp": row[0], "commit": row[1], "duration": row[2]}
                for row in cursor.fetchall()
            ]
    
    def get_performance_trends(self, days: int = 30) -> Dict:
        """Get performance trend analysis."""
        cutoff = time.time() - (days * 24 * 3600)
        
        with sqlite3.connect(self.db_path) as conn:
            # Overall trends
            cursor = conn.execute("""
                SELECT 
                    AVG(total_time) as avg_total_time,
                    AVG(average_time) as avg_test_time,
                    AVG(success_rate) as avg_success_rate,
                    COUNT(*) as run_count
                FROM test_runs 
                WHERE timestamp > ?
            """, (cutoff,))
            
            overall = dict(zip([desc[0] for desc in cursor.description], cursor.fetchone()))
            
            # Slowest tests
            cursor = conn.execute("""
                SELECT tt.test_name, AVG(tt.duration) as avg_duration, COUNT(*) as run_count
                FROM test_timings tt
                JOIN test_runs tr ON tt.run_id = tr.id
                WHERE tr.timestamp > ?
                GROUP BY tt.test_name
                HAVING COUNT(*) >= 3
                ORDER BY avg_duration DESC
                LIMIT 10
            """, (cutoff,))
            
            slowest_tests = [
                {"test": row[0], "avg_duration": row[1], "run_count": row[2]}
                for row in cursor.fetchall()
            ]
            
            return {
                "overall": overall,
                "slowest_tests": slowest_tests,
            }


class PerformanceRegressionDetector:
    """Detect performance regressions in test execution."""
    
    def __init__(self, db: TestPerformanceDatabase):
        self.db = db
        self.thresholds = {
            "duration_increase_percent": 50.0,  # 50% increase threshold
            "total_time_increase_percent": 25.0,  # 25% total time increase
            "minimum_runs": 5,  # Minimum runs to establish baseline
        }
    
    def detect_regressions(self, current_results: Dict) -> List[Dict]:
        """Detect performance regressions in current test run."""
        regressions = []
        
        # Get baseline performance (average of last 10 runs, excluding current)
        recent_runs = self.db.get_recent_runs(days=14)
        if len(recent_runs) < self.thresholds["minimum_runs"]:
            return regressions  # Not enough data
        
        # Calculate baseline total time
        baseline_total_times = [run["total_time"] for run in recent_runs[:10]]
        baseline_avg_total = sum(baseline_total_times) / len(baseline_total_times)
        
        # Check total time regression
        current_total = current_results["total_time"]
        total_increase = ((current_total - baseline_avg_total) / baseline_avg_total) * 100
        
        if total_increase > self.thresholds["total_time_increase_percent"]:
            regressions.append({
                "type": "total_time_regression",
                "current": current_total,
                "baseline": baseline_avg_total,
                "increase_percent": total_increase,
                "message": f"Total execution time increased by {total_increase:.1f}%",
            })
        
        # Check individual test regressions
        if "timings" in current_results:
            for timing in current_results["timings"]:
                test_history = self.db.get_test_history(timing["test"], days=14)
                
                if len(test_history) >= self.thresholds["minimum_runs"]:
                    baseline_durations = [h["duration"] for h in test_history[:10]]
                    baseline_avg = sum(baseline_durations) / len(baseline_durations)
                    
                    current_duration = timing["duration"]
                    increase = ((current_duration - baseline_avg) / baseline_avg) * 100
                    
                    if increase > self.thresholds["duration_increase_percent"]:
                        regressions.append({
                            "type": "test_duration_regression",
                            "test": timing["test"],
                            "current": current_duration,
                            "baseline": baseline_avg,
                            "increase_percent": increase,
                            "message": f"Test {timing['test']} duration increased by {increase:.1f}%",
                        })
        
        return regressions
    
    def record_regressions(self, regressions: List[Dict]):
        """Record detected regressions as alerts."""
        for regression in regressions:
            self.db.record_alert(
                alert_type=regression["type"],
                test_name=regression.get("test", "ALL"),
                current=regression["current"],
                baseline=regression["baseline"],
                threshold=regression["increase_percent"],
                message=regression["message"],
            )


class PerformanceDashboard:
    """Generate performance monitoring dashboard."""
    
    def __init__(self, db: TestPerformanceDatabase):
        self.db = db
    
    def generate_html_dashboard(self, output_file: str = "test_performance_dashboard.html"):
        """Generate HTML performance dashboard."""
        trends = self.db.get_performance_trends()
        recent_runs = self.db.get_recent_runs(days=7)
        
        html_content = self._create_html_template()
        
        # Insert data
        html_content = html_content.replace("{{TRENDS_DATA}}", json.dumps(trends))
        html_content = html_content.replace("{{RECENT_RUNS}}", json.dumps(recent_runs))
        html_content = html_content.replace("{{LAST_UPDATED}}", datetime.now().isoformat())
        
        Path(output_file).write_text(html_content)
        print(f"📊 Dashboard generated: {output_file}")
    
    def _create_html_template(self) -> str:
        """Create HTML dashboard template."""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Test Performance Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { text-align: center; margin-bottom: 30px; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .metric-card { border: 1px solid #ddd; border-radius: 8px; padding: 20px; text-align: center; }
        .metric-value { font-size: 2em; font-weight: bold; color: #007bff; }
        .metric-label { font-size: 0.9em; color: #666; margin-top: 5px; }
        .chart-container { margin: 20px 0; height: 400px; }
        .table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        .table th, .table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        .table th { background-color: #f2f2f2; }
        .status-good { color: #28a745; }
        .status-warning { color: #ffc107; }
        .status-danger { color: #dc3545; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Test Performance Dashboard</h1>
        <p>Last Updated: {{LAST_UPDATED}}</p>
    </div>
    
    <div class="metrics" id="metricsContainer">
        <!-- Metrics will be populated by JavaScript -->
    </div>
    
    <div class="chart-container">
        <canvas id="performanceChart"></canvas>
    </div>
    
    <h2>📋 Recent Test Runs</h2>
    <table class="table" id="recentRunsTable">
        <thead>
            <tr>
                <th>Timestamp</th>
                <th>Total Tests</th>
                <th>Total Time</th>
                <th>Average Time</th>
                <th>Success Rate</th>
                <th>Status</th>
            </tr>
        </thead>
        <tbody id="recentRunsBody">
            <!-- Runs will be populated by JavaScript -->
        </tbody>
    </table>
    
    <h2>🐌 Slowest Tests</h2>
    <table class="table" id="slowestTestsTable">
        <thead>
            <tr>
                <th>Test Name</th>
                <th>Average Duration</th>
                <th>Run Count</th>
                <th>Status</th>
            </tr>
        </thead>
        <tbody id="slowestTestsBody">
            <!-- Tests will be populated by JavaScript -->
        </tbody>
    </table>
    
    <script>
        const trendsData = {{TRENDS_DATA}};
        const recentRuns = {{RECENT_RUNS}};
        
        // Populate metrics
        function populateMetrics() {
            const metricsContainer = document.getElementById('metricsContainer');
            const overall = trendsData.overall;
            
            const metrics = [
                { label: 'Average Total Time', value: `${overall.avg_total_time?.toFixed(1) || 0}s`, status: overall.avg_total_time > 300 ? 'danger' : 'good' },
                { label: 'Average Test Time', value: `${(overall.avg_test_time * 1000)?.toFixed(0) || 0}ms`, status: overall.avg_test_time > 0.5 ? 'warning' : 'good' },
                { label: 'Success Rate', value: `${(overall.avg_success_rate * 100)?.toFixed(1) || 0}%`, status: overall.avg_success_rate < 0.95 ? 'danger' : 'good' },
                { label: 'Total Runs', value: overall.run_count || 0, status: 'good' }
            ];
            
            metrics.forEach(metric => {
                const card = document.createElement('div');
                card.className = 'metric-card';
                card.innerHTML = `
                    <div class="metric-value status-${metric.status}">${metric.value}</div>
                    <div class="metric-label">${metric.label}</div>
                `;
                metricsContainer.appendChild(card);
            });
        }
        
        // Populate recent runs table
        function populateRecentRuns() {
            const tbody = document.getElementById('recentRunsBody');
            
            recentRuns.forEach(run => {
                const row = document.createElement('tr');
                const date = new Date(run.timestamp * 1000).toLocaleString();
                const status = run.success_rate >= 0.95 ? 'good' : run.success_rate >= 0.8 ? 'warning' : 'danger';
                
                row.innerHTML = `
                    <td>${date}</td>
                    <td>${run.total_tests}</td>
                    <td>${run.total_time.toFixed(1)}s</td>
                    <td>${(run.average_time * 1000).toFixed(0)}ms</td>
                    <td class="status-${status}">${(run.success_rate * 100).toFixed(1)}%</td>
                    <td class="status-${status}">●</td>
                `;
                tbody.appendChild(row);
            });
        }
        
        // Populate slowest tests table
        function populateSlowestTests() {
            const tbody = document.getElementById('slowestTestsBody');
            
            trendsData.slowest_tests.forEach(test => {
                const row = document.createElement('tr');
                const status = test.avg_duration > 2 ? 'danger' : test.avg_duration > 0.5 ? 'warning' : 'good';
                
                row.innerHTML = `
                    <td>${test.test}</td>
                    <td class="status-${status}">${test.avg_duration.toFixed(3)}s</td>
                    <td>${test.run_count}</td>
                    <td class="status-${status}">●</td>
                `;
                tbody.appendChild(row);
            });
        }
        
        // Create performance chart
        function createPerformanceChart() {
            const ctx = document.getElementById('performanceChart').getContext('2d');
            
            const chartData = {
                labels: recentRuns.map(run => new Date(run.timestamp * 1000).toLocaleDateString()).reverse(),
                datasets: [{
                    label: 'Total Execution Time (s)',
                    data: recentRuns.map(run => run.total_time).reverse(),
                    borderColor: 'rgb(75, 192, 192)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    tension: 0.1
                }, {
                    label: 'Average Test Time (ms)',
                    data: recentRuns.map(run => run.average_time * 1000).reverse(),
                    borderColor: 'rgb(255, 99, 132)',
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    tension: 0.1,
                    yAxisID: 'y1'
                }]
            };
            
            new Chart(ctx, {
                type: 'line',
                data: chartData,
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Test Performance Trends'
                        }
                    },
                    scales: {
                        y: {
                            type: 'linear',
                            display: true,
                            position: 'left',
                            title: {
                                display: true,
                                text: 'Total Time (s)'
                            }
                        },
                        y1: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            title: {
                                display: true,
                                text: 'Average Time (ms)'
                            },
                            grid: {
                                drawOnChartArea: false,
                            },
                        }
                    }
                }
            });
        }
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            populateMetrics();
            populateRecentRuns();
            populateSlowestTests();
            createPerformanceChart();
        });
    </script>
</body>
</html>
        """
    
    def generate_performance_report(self) -> str:
        """Generate text-based performance report."""
        trends = self.db.get_performance_trends()
        recent_runs = self.db.get_recent_runs(days=7)
        
        report = []
        report.append("🚀 Test Performance Report")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("")
        
        # Overall trends
        overall = trends["overall"]
        report.append("📊 Performance Overview:")
        report.append(f"  Average Total Time: {overall['avg_total_time']:.1f}s")
        report.append(f"  Average Test Time: {overall['avg_test_time']*1000:.0f}ms")
        report.append(f"  Success Rate: {overall['avg_success_rate']*100:.1f}%")
        report.append(f"  Total Runs Analyzed: {overall['run_count']}")
        report.append("")
        
        # Performance status
        report.append("🎯 Performance Status:")
        
        if overall["avg_total_time"] > 300:
            report.append("  ❌ Total execution time is high (>5min)")
        elif overall["avg_total_time"] > 180:
            report.append("  ⚠️  Total execution time is moderate (>3min)")
        else:
            report.append("  ✅ Total execution time is good (<3min)")
        
        if overall["avg_test_time"] > 0.5:
            report.append("  ❌ Average test time is high (>500ms)")
        elif overall["avg_test_time"] > 0.1:
            report.append("  ⚠️  Average test time is moderate (>100ms)")
        else:
            report.append("  ✅ Average test time is good (<100ms)")
        
        if overall["avg_success_rate"] < 0.95:
            report.append("  ❌ Success rate is low (<95%)")
        else:
            report.append("  ✅ Success rate is good (≥95%)")
        
        report.append("")
        
        # Slowest tests
        report.append("🐌 Top 5 Slowest Tests:")
        for i, test in enumerate(trends["slowest_tests"][:5], 1):
            status = "❌" if test["avg_duration"] > 2 else "⚠️" if test["avg_duration"] > 0.5 else "✅"
            report.append(f"  {i}. {status} {test['test']} - {test['avg_duration']:.3f}s ({test['run_count']} runs)")
        
        report.append("")
        
        # Recent performance
        if recent_runs:
            latest = recent_runs[0]
            report.append("📈 Latest Run Performance:")
            report.append(f"  Timestamp: {datetime.fromtimestamp(latest['timestamp'])}")
            report.append(f"  Total Tests: {latest['total_tests']}")
            report.append(f"  Total Time: {latest['total_time']:.1f}s")
            report.append(f"  Average Time: {latest['average_time']*1000:.0f}ms")
            report.append(f"  Success Rate: {latest['success_rate']*100:.1f}%")
        
        return "\n".join(report)


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test performance monitoring dashboard")
    parser.add_argument("--db", default="test_performance.db", help="Database file path")
    parser.add_argument("--dashboard", action="store_true", help="Generate HTML dashboard")
    parser.add_argument("--report", action="store_true", help="Generate text report")
    parser.add_argument("--days", type=int, default=30, help="Days of history to analyze")
    
    args = parser.parse_args()
    
    db = TestPerformanceDatabase(args.db)
    dashboard = PerformanceDashboard(db)
    
    if args.dashboard:
        dashboard.generate_html_dashboard()
    
    if args.report:
        report = dashboard.generate_performance_report()
        print(report)
        
        # Save report to file
        report_file = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        Path(report_file).write_text(report)
        print(f"\n📄 Report saved to: {report_file}")
    
    if not args.dashboard and not args.report:
        # Default: show quick status
        trends = db.get_performance_trends(args.days)
        overall = trends["overall"]
        
        print("🚀 Quick Performance Status:")
        print(f"  Total Time: {overall['avg_total_time']:.1f}s")
        print(f"  Test Time: {overall['avg_test_time']*1000:.0f}ms")
        print(f"  Success Rate: {overall['avg_success_rate']*100:.1f}%")
        print(f"  Runs: {overall['run_count']}")


if __name__ == "__main__":
    main()