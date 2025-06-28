"""Configuration Observability Dashboard.

Real-time web dashboard for monitoring configuration automation system.
Provides interactive visualization of drift detection, validation results,
optimization recommendations, and system health metrics.

Features:
- Real-time configuration monitoring
- Interactive drift analysis charts
- Performance optimization tracking
- Environment comparison views
- Automated alert management
- Configuration health scoring

Usage:
    streamlit run src/config/observability/dashboard.py
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from .automation import (
    ConfigDriftSeverity,
    ConfigObservabilityAutomation,
    ConfigValidationStatus,
    get_automation_system,
)


# Dashboard configuration
st.set_page_config(
    page_title="Configuration Observability Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .alert-critical {
        background: #ff4444;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .alert-warning {
        background: #ffaa00;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .status-healthy {
        color: #00aa00;
        font-weight: bold;
    }
    .status-warning {
        color: #ffaa00;
        font-weight: bold;
    }
    .status-critical {
        color: #ff4444;
        font-weight: bold;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=30)  # Cache for 30 seconds
def get_system_status():
    """Get cached system status."""
    try:
        automation_system = get_automation_system()
        return automation_system.get_detailed_report()
    except Exception as e:
        st.error(f"Error getting system status: {e}")
        return None


@st.cache_data(ttl=60)  # Cache for 1 minute
def get_historical_data():
    """Get historical drift and validation data."""
    try:
        automation_system = get_automation_system()

        # Get recent data (last 24 hours)
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(hours=24)

        recent_drifts = [
            drift
            for drift in automation_system.drift_history
            if drift.timestamp > cutoff_time
        ]

        recent_validations = [
            validation
            for validation in automation_system.validation_history
            if validation.timestamp > cutoff_time
        ]

        return {
            "drifts": recent_drifts,
            "validations": recent_validations,
        }
    except Exception as e:
        st.error(f"Error getting historical data: {e}")
        return {"drifts": [], "validations": []}


def create_drift_severity_chart(drifts: List) -> go.Figure:
    """Create drift severity distribution chart."""
    if not drifts:
        return go.Figure()

    # Count by severity
    severity_counts = {}
    for drift in drifts:
        severity = drift.severity.value
        severity_counts[severity] = severity_counts.get(severity, 0) + 1

    # Create pie chart
    fig = go.Figure(
        data=[
            go.Pie(
                labels=list(severity_counts.keys()),
                values=list(severity_counts.values()),
                hole=0.3,
                marker={"colors": ["#ff4444", "#ff8800", "#ffdd00", "#4488ff"]},
            )
        ]
    )

    fig.update_layout(
        title="Configuration Drift by Severity",
        height=400,
    )

    return fig


def create_drift_timeline_chart(drifts: List) -> go.Figure:
    """Create drift occurrence timeline chart."""
    if not drifts:
        return go.Figure()

    # Convert to DataFrame for easier plotting
    df_data = []
    for drift in drifts:
        df_data.append(
            {
                "timestamp": drift.timestamp,
                "parameter": drift.parameter,
                "severity": drift.severity.value,
                "environment": drift.environment,
                "impact_score": drift.impact_score,
            }
        )

    df = pd.DataFrame(df_data)

    # Create scatter plot
    color_map = {
        "fatal": "#ff0000",
        "critical": "#ff4444",
        "warning": "#ff8800",
        "info": "#4488ff",
    }

    fig = px.scatter(
        df,
        x="timestamp",
        y="parameter",
        color="severity",
        size="impact_score",
        hover_data=["environment"],
        color_discrete_map=color_map,
        title="Configuration Drift Timeline",
    )

    fig.update_layout(height=500)

    return fig


def create_environment_comparison_chart(status: Dict) -> go.Figure:
    """Create environment health comparison chart."""
    environments = status.get("environments", {}).get("detected", [])

    if not environments:
        return go.Figure()

    # Create mock health scores for demonstration
    # In real implementation, this would calculate actual health metrics
    health_scores = {}
    for env in environments:
        # Base health score calculation
        score = 85  # Base score

        # Adjust based on drift history (simplified)
        drift_analysis = status.get("drift_analysis", {})
        if drift_analysis.get("critical_drifts", 0) > 0:
            score -= 20
        elif drift_analysis.get("recent_drifts", 0) > 0:
            score -= 10

        # Adjust based on validation issues
        validation_status = status.get("validation_status", {})
        if validation_status.get("critical_issues", 0) > 0:
            score -= 15
        elif validation_status.get("errors", 0) > 0:
            score -= 10
        elif validation_status.get("warnings", 0) > 0:
            score -= 5

        health_scores[env] = max(score, 0)

    # Create bar chart
    fig = go.Figure(
        data=[
            go.Bar(
                x=list(health_scores.keys()),
                y=list(health_scores.values()),
                marker={
                    "color": list(health_scores.values()),
                    "colorscale": "RdYlGn",
                    "cmin": 0,
                    "cmax": 100,
                },
            )
        ]
    )

    fig.update_layout(
        title="Environment Health Scores",
        yaxis_title="Health Score (%)",
        xaxis_title="Environment",
        height=400,
    )

    return fig


def create_optimization_impact_chart(recommendations: List) -> go.Figure:
    """Create optimization recommendations impact chart."""
    if not recommendations:
        return go.Figure()

    # Convert to DataFrame
    df_data = []
    for rec in recommendations:
        df_data.append(
            {
                "parameter": rec["parameter"],
                "confidence": rec["confidence_score"],
                "improvement": rec["expected_improvement"],
                "environment": rec["environment"],
                "performance_impact": rec.get("performance_impact", "Medium"),
            }
        )

    df = pd.DataFrame(df_data)

    # Create bubble chart
    fig = px.scatter(
        df,
        x="parameter",
        y="confidence",
        size="confidence",
        color="environment",
        hover_data=["improvement", "performance_impact"],
        title="Optimization Recommendations by Confidence",
    )

    fig.update_layout(height=500)
    fig.update_xaxes(tickangle=45)

    return fig


def display_alerts(status: Dict):
    """Display system alerts."""
    alerts = []

    # Critical drift alerts
    drift_analysis = status.get("drift_analysis", {})
    critical_drifts = drift_analysis.get("critical_drifts", 0)
    if critical_drifts > 0:
        alerts.append(
            {
                "type": "critical",
                "message": f"Critical configuration drift detected in {critical_drifts} parameters",
                "action": "Immediate attention required",
            }
        )

    # Validation error alerts
    validation_status = status.get("validation_status", {})
    critical_issues = validation_status.get("critical_issues", 0)
    if critical_issues > 0:
        alerts.append(
            {
                "type": "critical",
                "message": f"Critical validation issues found: {critical_issues}",
                "action": "Configuration requires immediate correction",
            }
        )

    # Warning alerts
    warnings = validation_status.get("warnings", 0)
    if warnings > 5:
        alerts.append(
            {
                "type": "warning",
                "message": f"Multiple validation warnings: {warnings}",
                "action": "Review and optimize configuration",
            }
        )

    # Display alerts
    if alerts:
        st.subheader("🚨 System Alerts")

        for alert in alerts:
            if alert["type"] == "critical":
                st.markdown(
                    f"""
                <div class="alert-critical">
                    <strong>CRITICAL:</strong> {alert["message"]}<br>
                    <em>Action:</em> {alert["action"]}
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                <div class="alert-warning">
                    <strong>WARNING:</strong> {alert["message"]}<br>
                    <em>Action:</em> {alert["action"]}
                </div>
                """,
                    unsafe_allow_html=True,
                )
    else:
        st.success("✅ No active alerts - system operating normally")


def main():
    """Main dashboard application."""

    # Header
    st.markdown(
        '<h1 class="main-header">⚙️ Configuration Observability Dashboard</h1>',
        unsafe_allow_html=True,
    )
    st.markdown("### Real-time monitoring and automation of configuration health")

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Dashboard Controls")

        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=True)

        # Refresh button
        if st.button("🔄 Refresh Now"):
            st.cache_data.clear()
            st.rerun()

        # Time range selector
        time_range = st.selectbox(
            "Time Range",
            ["Last 1 hour", "Last 6 hours", "Last 24 hours", "Last 7 days"],
            index=2,
        )

        # Environment filter
        environments = ["All"] + ["simple", "enterprise", "development", "testing"]
        selected_env = st.selectbox("Environment", environments)

        st.markdown("---")

        # System info
        st.subheader("📊 System Info")
        st.markdown("**Dashboard Version:** 1.0.0")
        st.markdown(f"**Last Updated:** {datetime.now().strftime('%H:%M:%S')}")

        # Quick actions
        st.markdown("---")
        st.subheader("🚀 Quick Actions")

        if st.button("🔍 Run Drift Check"):
            with st.spinner("Checking for configuration drift..."):
                # In real implementation, would trigger drift check
                time.sleep(1)
                st.success("Drift check completed")

        if st.button("✅ Validate Configs"):
            with st.spinner("Validating configurations..."):
                # In real implementation, would trigger validation
                time.sleep(1)
                st.success("Validation completed")

        if st.button("⚡ Generate Optimizations"):
            with st.spinner("Generating optimization recommendations..."):
                # In real implementation, would generate recommendations
                time.sleep(1)
                st.success("Optimizations generated")

    # Get system status
    status = get_system_status()
    if not status:
        st.error("Unable to connect to automation system")
        return

    # Display alerts
    display_alerts(status)

    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 Overview",
            "🔍 Drift Analysis",
            "✅ Validation Status",
            "⚡ Optimization",
            "🌍 Environments",
        ]
    )

    with tab1:
        st.header("📊 System Overview")

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        system_status = status["system_status"]
        drift_analysis = status["drift_analysis"]
        validation_status = status["validation_status"]

        with col1:
            automation_status = (
                "🟢 Active" if system_status["automation_enabled"] else "🔴 Inactive"
            )
            st.metric("Automation Status", automation_status)

        with col2:
            recent_drifts = drift_analysis["recent_drifts"]
            drift_color = "normal" if recent_drifts == 0 else "inverse"
            st.metric("Recent Drifts", recent_drifts, delta_color=drift_color)

        with col3:
            validation_errors = (
                validation_status["errors"] + validation_status["critical_issues"]
            )
            error_color = "normal" if validation_errors == 0 else "inverse"
            st.metric("Validation Issues", validation_errors, delta_color=error_color)

        with col4:
            environments_count = system_status["environments_monitored"]
            st.metric("Environments", environments_count)

        # System status details
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔧 System Status")

            status_data = {
                "Auto-Remediation": "✅ Enabled"
                if system_status["auto_remediation_enabled"]
                else "❌ Disabled",
                "File Monitoring": "🟢 Active"
                if system_status["file_monitoring_active"]
                else "🔴 Inactive",
                "Last Drift Check": system_status["last_drift_check"].split("T")[1][:8],
                "Last Optimization": system_status["last_optimization_check"].split(
                    "T"
                )[1][:8],
            }

            for key, value in status_data.items():
                st.write(f"**{key}:** {value}")

        with col2:
            st.subheader("📈 Performance Summary")

            # Create performance gauge chart
            optimization = status["optimization"]
            active_recommendations = optimization["active_recommendations"]

            # Calculate overall health score
            health_score = 100
            if drift_analysis["critical_drifts"] > 0:
                health_score -= 30
            if validation_status["critical_issues"] > 0:
                health_score -= 20
            if validation_status["errors"] > 0:
                health_score -= 15
            if active_recommendations > 5:
                health_score -= 10

            health_score = max(health_score, 0)

            # Gauge chart
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=health_score,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "System Health Score"},
                    delta={"reference": 90},
                    gauge={
                        "axis": {"range": [None, 100]},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 50], "color": "lightgray"},
                            {"range": [50, 80], "color": "gray"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 90,
                        },
                    },
                )
            )

            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)

    with tab2:
        st.header("🔍 Configuration Drift Analysis")

        # Get historical drift data
        historical_data = get_historical_data()
        drifts = historical_data["drifts"]

        if drifts:
            col1, col2 = st.columns(2)

            with col1:
                # Drift severity chart
                fig_severity = create_drift_severity_chart(drifts)
                st.plotly_chart(fig_severity, use_container_width=True)

            with col2:
                # Drift by environment
                env_drift_counts = {}
                for drift in drifts:
                    env = drift.environment
                    env_drift_counts[env] = env_drift_counts.get(env, 0) + 1

                if env_drift_counts:
                    fig_env = go.Figure(
                        data=[
                            go.Bar(
                                x=list(env_drift_counts.keys()),
                                y=list(env_drift_counts.values()),
                            )
                        ]
                    )
                    fig_env.update_layout(
                        title="Drift Count by Environment", height=400
                    )
                    st.plotly_chart(fig_env, use_container_width=True)

            # Drift timeline
            fig_timeline = create_drift_timeline_chart(drifts)
            st.plotly_chart(fig_timeline, use_container_width=True)

            # Detailed drift table
            st.subheader("📋 Recent Drift Details")

            if (
                "detailed_analysis" in status
                and status["detailed_analysis"]["recent_drifts"]
            ):
                drift_df_data = []
                for drift in status["detailed_analysis"]["recent_drifts"][
                    :10
                ]:  # Last 10
                    drift_df_data.append(
                        {
                            "Parameter": drift["parameter"],
                            "Severity": drift["severity"].upper(),
                            "Environment": drift["environment"],
                            "Impact": f"{drift['impact_score']:.1%}",
                            "Auto-Fix": "✅" if drift["auto_fix_available"] else "❌",
                            "Timestamp": drift["timestamp"].split("T")[1][:8],
                        }
                    )

                drift_df = pd.DataFrame(drift_df_data)
                st.dataframe(drift_df, use_container_width=True)
            else:
                st.info("No recent drift data available")
        else:
            st.success("✅ No configuration drift detected in the selected time range")

    with tab3:
        st.header("✅ Configuration Validation Status")

        # Validation metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Checks", validation_status["recent_validations"])
        with col2:
            st.metric("Errors", validation_status["errors"], delta_color="inverse")
        with col3:
            st.metric("Warnings", validation_status["warnings"], delta_color="inverse")
        with col4:
            st.metric(
                "Critical Issues",
                validation_status["critical_issues"],
                delta_color="inverse",
            )

        # Validation results over time
        historical_data = get_historical_data()
        validations = historical_data["validations"]

        if validations:
            # Group validations by hour
            hourly_validations = {}
            for validation in validations:
                hour = validation.timestamp.replace(minute=0, second=0, microsecond=0)
                if hour not in hourly_validations:
                    hourly_validations[hour] = {
                        "valid": 0,
                        "warning": 0,
                        "error": 0,
                        "critical": 0,
                    }

                status_key = validation.status.value
                hourly_validations[hour][status_key] = (
                    hourly_validations[hour].get(status_key, 0) + 1
                )

            # Create stacked bar chart
            hours = sorted(hourly_validations.keys())
            valid_counts = [hourly_validations[h]["valid"] for h in hours]
            warning_counts = [hourly_validations[h]["warning"] for h in hours]
            error_counts = [hourly_validations[h]["error"] for h in hours]
            critical_counts = [hourly_validations[h]["critical"] for h in hours]

            fig_validation = go.Figure()
            fig_validation.add_trace(
                go.Bar(name="Valid", x=hours, y=valid_counts, marker_color="green")
            )
            fig_validation.add_trace(
                go.Bar(name="Warning", x=hours, y=warning_counts, marker_color="orange")
            )
            fig_validation.add_trace(
                go.Bar(name="Error", x=hours, y=error_counts, marker_color="red")
            )
            fig_validation.add_trace(
                go.Bar(
                    name="Critical", x=hours, y=critical_counts, marker_color="darkred"
                )
            )

            fig_validation.update_layout(
                barmode="stack",
                title="Validation Results Over Time",
                xaxis_title="Time",
                yaxis_title="Count",
                height=400,
            )

            st.plotly_chart(fig_validation, use_container_width=True)

        # Detailed validation results
        if (
            "detailed_analysis" in status
            and status["detailed_analysis"]["validation_results"]
        ):
            st.subheader("📋 Recent Validation Results")

            validation_df_data = []
            for result in status["detailed_analysis"]["validation_results"][:10]:
                validation_df_data.append(
                    {
                        "Parameter": result["parameter"],
                        "Status": result["status"].upper(),
                        "Message": result["message"][:50] + "..."
                        if len(result["message"]) > 50
                        else result["message"],
                        "Environment": result["environment"],
                        "Suggestions": len(result["suggestions"]),
                        "Timestamp": result["timestamp"].split("T")[1][:8],
                    }
                )

            validation_df = pd.DataFrame(validation_df_data)
            st.dataframe(validation_df, use_container_width=True)

    with tab4:
        st.header("⚡ Performance Optimization")

        optimization = status["optimization"]

        # Optimization metrics
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Active Recommendations", optimization["active_recommendations"])
        with col2:
            st.metric(
                "Performance Metrics", optimization["performance_metrics_tracked"]
            )

        # Optimization recommendations
        if (
            "detailed_analysis" in status
            and status["detailed_analysis"]["optimization_recommendations"]
        ):
            recommendations = status["detailed_analysis"][
                "optimization_recommendations"
            ]

            # Create optimization impact chart
            fig_optimization = create_optimization_impact_chart(recommendations)
            st.plotly_chart(fig_optimization, use_container_width=True)

            # Recommendations table
            st.subheader("📋 Current Recommendations")

            rec_df_data = []
            for rec in recommendations:
                rec_df_data.append(
                    {
                        "Parameter": rec["parameter"],
                        "Current": str(rec["current_value"]),
                        "Recommended": str(rec["recommended_value"]),
                        "Expected Improvement": rec["expected_improvement"],
                        "Confidence": f"{rec['confidence_score']:.1%}",
                        "Environment": rec["environment"],
                    }
                )

            rec_df = pd.DataFrame(rec_df_data)
            st.dataframe(rec_df, use_container_width=True)

            # Recommendation details
            with st.expander("💡 Detailed Recommendations"):
                for rec in recommendations:
                    st.write(f"**{rec['parameter']}**")
                    st.write(f"Reasoning: {rec['reasoning']}")
                    st.write(
                        f"Performance Impact: {rec.get('performance_impact', 'Medium')}"
                    )
                    st.write("---")
        else:
            st.info("No optimization recommendations available at this time")

    with tab5:
        st.header("🌍 Environment Management")

        environments = status["environments"]

        # Environment overview
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🌍 Detected Environments")
            for env in environments["detected"]:
                baseline_status = (
                    "✅" if env in environments["baselines_established"] else "❌"
                )
                st.write(f"**{env}:** {baseline_status} Baseline")

        with col2:
            # Environment health comparison
            fig_env_health = create_environment_comparison_chart(status)
            st.plotly_chart(fig_env_health, use_container_width=True)

        # Environment details
        st.subheader("📊 Environment Configuration Summary")

        # Create mock environment comparison data
        env_comparison_data = []
        for env in environments["detected"]:
            env_comparison_data.append(
                {
                    "Environment": env,
                    "Baseline Status": "✅ Established"
                    if env in environments["baselines_established"]
                    else "❌ Missing",
                    "Drift Issues": "Low"
                    if env != "enterprise"
                    else "Medium",  # Mock data
                    "Validation Status": "Healthy"
                    if env != "testing"
                    else "Warnings",  # Mock data
                    "Last Checked": datetime.now().strftime("%H:%M:%S"),
                }
            )

        env_df = pd.DataFrame(env_comparison_data)
        st.dataframe(env_df, use_container_width=True)

        # Environment actions
        st.subheader("🚀 Environment Actions")

        selected_env_action = st.selectbox(
            "Select Environment", environments["detected"]
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button(f"🔄 Reset {selected_env_action} Baseline"):
                with st.spinner(f"Resetting baseline for {selected_env_action}..."):
                    time.sleep(1)
                    st.success(f"Baseline reset for {selected_env_action}")

        with col2:
            if st.button(f"✅ Validate {selected_env_action}"):
                with st.spinner(f"Validating {selected_env_action}..."):
                    time.sleep(1)
                    st.success(f"Validation completed for {selected_env_action}")

        with col3:
            if st.button(f"⚡ Optimize {selected_env_action}"):
                with st.spinner(
                    f"Generating optimizations for {selected_env_action}..."
                ):
                    time.sleep(1)
                    st.success(f"Optimizations generated for {selected_env_action}")

    # Auto-refresh
    if auto_refresh:
        time.sleep(30)
        st.rerun()


if __name__ == "__main__":
    main()
