
"""Configuration audit and change logging for wizard operations.

Provides audit logging for configuration changes, validation history,
and security tracking for wizard-generated configurations.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()


class ConfigAuditor:
    """Audits and logs configuration changes made through the wizard."""

    def __init__(self, audit_dir: Path | None = None):
        """Initialize configuration auditor.

        Args:
            audit_dir: Directory for storing audit logs
        """
        self.audit_dir = audit_dir or Path("config/audit")
        self.audit_dir.mkdir(parents=True, exist_ok=True)

        # Setup audit logging
        self.audit_log = self.audit_dir / "wizard_audit.log"
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Setup audit logger."""
        logger = logging.getLogger("wizard_audit")
        logger.setLevel(logging.INFO)

        # Avoid duplicate handlers
        if logger.handlers:
            return logger

        # File handler for audit log
        handler = logging.FileHandler(self.audit_log)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def log_wizard_start(self, profile: str, template: str) -> None:
        """Log the start of a wizard session."""
        self.logger.info(
            f"Wizard session started - Profile: {profile}, Template: {template}"
        )

    def log_wizard_completion(
        self, profile: str, config_path: Path, customizations: dict[str, Any]
    ) -> None:
        """Log successful wizard completion."""
        self.logger.info(
            f"Wizard completed - Profile: {profile}, Config: {config_path}, "
            f"Customizations: {len(customizations)} items"
        )

        # Save detailed audit record
        audit_record = {
            "timestamp": datetime.now().isoformat(),
            "action": "wizard_completion",
            "profile": profile,
            "config_path": str(config_path),
            "customizations": customizations,
        }

        self._save_audit_record(audit_record)

    def log_validation_failure(self, errors: list[str]) -> None:
        """Log configuration validation failures."""
        self.logger.warning(f"Validation failed - {len(errors)} errors: {errors}")

    def log_security_event(self, event_type: str, details: str) -> None:
        """Log security-related events."""
        self.logger.warning(f"Security event - {event_type}: {details}")

    def log_template_customization(
        self, template: str, section: str, changes: dict[str, Any]
    ) -> None:
        """Log template customizations."""
        self.logger.info(
            f"Template customized - Template: {template}, Section: {section}, "
            f"Changes: {changes}"
        )

    def _save_audit_record(self, record: dict[str, Any]) -> None:
        """Save detailed audit record to JSON file."""
        audit_file = self.audit_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.json"

        # Load existing records
        records = []
        if audit_file.exists():
            try:
                with open(audit_file) as f:
                    records = json.load(f)
            except (OSError, json.JSONDecodeError):
                records = []

        # Add new record
        records.append(record)

        # Save back to file
        try:
            with open(audit_file, "w") as f:
                json.dump(records, f, indent=2)
        except OSError as e:
            self.logger.exception(f"Failed to save audit record: {e}")

    def get_recent_activity(self, days: int = 7) -> list[dict[str, Any]]:
        """Get recent wizard activity from audit logs.

        Args:
            days: Number of days to look back

        Returns:
            List of recent audit records
        """
        recent_records = []

        # Check audit files from the last N days
        from datetime import timedelta

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        current_date = start_date
        while current_date <= end_date:
            audit_file = (
                self.audit_dir / f"audit_{current_date.strftime('%Y%m%d')}.json"
            )

            if audit_file.exists():
                try:
                    with open(audit_file) as f:
                        daily_records = json.load(f)
                        recent_records.extend(daily_records)
                except (OSError, json.JSONDecodeError):
                    continue

            current_date += timedelta(days=1)

        return sorted(recent_records, key=lambda x: x.get("timestamp", ""))

    def show_audit_summary(self) -> None:
        """Display a summary of recent wizard activity."""
        console.print("\n[bold cyan]🔍 Recent Wizard Activity[/bold cyan]")

        recent_activity = self.get_recent_activity(days=30)

        if not recent_activity:
            console.print("[dim]No recent wizard activity found.[/dim]")
            return

        # Summary statistics
        total_sessions = len(
            [r for r in recent_activity if r.get("action") == "wizard_completion"]
        )
        profiles_used = {r.get("profile") for r in recent_activity if r.get("profile")}

        console.print(f"• Total wizard sessions: {total_sessions}")
        console.print(f"• Profiles used: {', '.join(sorted(profiles_used))}")
        console.print(
            f"• Last activity: {recent_activity[-1].get('timestamp', 'Unknown') if recent_activity else 'None'}"
        )

        # Show recent sessions
        if total_sessions > 0:
            console.print("\n[bold]Recent Sessions:[/bold]")
            for record in recent_activity[-5:]:  # Last 5 sessions
                if record.get("action") == "wizard_completion":
                    timestamp = record.get("timestamp", "Unknown")
                    profile = record.get("profile", "Unknown")
                    config_path = record.get("config_path", "Unknown")
                    console.print(f"  • {timestamp[:19]} - {profile} → {config_path}")

    def cleanup_old_audits(self, keep_days: int = 90) -> None:
        """Clean up old audit files.

        Args:
            keep_days: Number of days of audit history to keep
        """
        from datetime import timedelta

        cutoff_date = datetime.now() - timedelta(days=keep_days)

        audit_files = list(self.audit_dir.glob("audit_*.json"))
        deleted_count = 0

        for audit_file in audit_files:
            try:
                # Extract date from filename
                date_str = audit_file.stem.replace("audit_", "")
                file_date = datetime.strptime(date_str, "%Y%m%d")

                if file_date < cutoff_date:
                    audit_file.unlink()
                    deleted_count += 1

            except (ValueError, OSError):
                # Skip files that don't match expected format or can't be deleted
                continue

        if deleted_count > 0:
            self.logger.info(f"Cleaned up {deleted_count} old audit files")
