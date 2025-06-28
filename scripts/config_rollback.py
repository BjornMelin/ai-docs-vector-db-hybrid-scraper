#!/usr/bin/env python3
"""
Configuration Rollback Script

This script provides automated rollback capabilities for configuration deployments,
supporting the GitOps configuration management workflow.
"""

import json  # noqa: PLC0415
import sys
import argparse
import logging  # noqa: PLC0415
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import tarfile

@dataclass
class DeploymentSnapshot:
    """Represents a configuration deployment snapshot"""
    snapshot_id: str
    environment: str
    timestamp: str
    commit_sha: str
    commit_message: str
    author: str
    deployment_strategy: str
    changed_files: List[str]
    
    @classmethod
    def from_file(cls, snapshot_file: Path) -> 'DeploymentSnapshot':
        """Load snapshot metadata from file"""
        with open(snapshot_file, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def to_file(self, snapshot_file: Path) -> None:
        """Save snapshot metadata to file"""
        with open(snapshot_file, 'w') as f:
            json.dump(asdict(self), f, indent=2)


class ConfigurationRollback:
    """
    Configuration rollback manager
    
    Provides capabilities for:
    - Listing available snapshots
    - Rolling back to previous configurations
    - Validating rollback targets
    - Generating rollback reports
    """
    
    def __init__(self, config_dir: Path, snapshots_dir: Path):
        self.config_dir = Path(config_dir)
        self.snapshots_dir = Path(snapshots_dir)
        self.logger = self._setup_logging()
        
        # Ensure directories exist
        self.snapshots_dir.mkdir(exist_ok=True)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger("config_rollback")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def list_snapshots(self, environment: Optional[str] = None) -> List[DeploymentSnapshot]:
        """List available deployment snapshots"""
        snapshots = []
        
        for snapshot_file in self.snapshots_dir.glob("*.json"):
            try:
                snapshot = DeploymentSnapshot.from_file(snapshot_file)
                
                if environment is None or snapshot.environment == environment:
                    snapshots.append(snapshot)
                    
            except Exception:
                self.logger.warning(f"Failed to load snapshot {snapshot_file}: {e}")
        
        # Sort by timestamp (newest first)
        snapshots.sort(key=lambda s: s.timestamp, reverse=True)
        return snapshots
    
    def get_snapshot(self, snapshot_id: str) -> Optional[DeploymentSnapshot]:
        """Get a specific snapshot by ID"""
        snapshot_file = self.snapshots_dir / f"{snapshot_id}.json"
        
        if not snapshot_file.exists():
            return None
        
        try:
            return DeploymentSnapshot.from_file(snapshot_file)
        except Exception:
            self.logger.error(f"Failed to load snapshot {snapshot_id}: {e}")
            return None
    
    def validate_rollback_target(self, snapshot: DeploymentSnapshot) -> bool:
        """Validate that a snapshot can be used for rollback"""
        snapshot_archive = self.snapshots_dir / f"{snapshot.snapshot_id}.tar.gz"
        
        if not snapshot_archive.exists():
            self.logger.error(f"Snapshot archive not found: {snapshot_archive}")
            return False
        
        # Verify archive integrity
        try:
            with tarfile.open(snapshot_archive, 'r:gz') as tar:
                tar.getnames()  # This will raise an exception if corrupted
            
            self.logger.info(f"Snapshot archive is valid: {snapshot_archive}")
            return True
            
        except Exception:
            self.logger.error(f"Snapshot archive is corrupted: {e}")
            return False
    
    def create_current_backup(self, environment: str) -> str:
        """Create a backup of current configuration before rollback"""
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")
        backup_id = f"pre-rollback-{environment}-{timestamp}"
        
        # Create backup archive
        backup_archive = self.snapshots_dir / f"{backup_id}.tar.gz"
        
        try:
            with tarfile.open(backup_archive, 'w:gz') as tar:
                tar.add(self.config_dir, arcname='config')
            
            # Create backup metadata
            backup_metadata = DeploymentSnapshot(
                snapshot_id=backup_id,
                environment=environment,
                timestamp=timestamp,
                commit_sha="pre-rollback",
                commit_message="Pre-rollback backup",
                author="config_rollback.py",
                deployment_strategy="backup",
                changed_files=[]
            )
            
            backup_metadata.to_file(self.snapshots_dir / f"{backup_id}.json")
            
            self.logger.info(f"Created pre-rollback backup: {backup_id}")
            return backup_id
            
        except Exception:
            self.logger.error(f"Failed to create backup: {e}")
            raise
    
    def restore_snapshot(self, snapshot: DeploymentSnapshot, create_backup: bool = True) -> bool:
        """Restore configuration from a snapshot"""
        self.logger.info(f"Starting rollback to snapshot: {snapshot.snapshot_id}")
        
        # Validate snapshot
        if not self.validate_rollback_target(snapshot):
            return False
        
        # Create backup if requested
        backup_id = None
        if create_backup:
            try:
                backup_id = self.create_current_backup(snapshot.environment)
            except Exception:
                self.logger.error(f"Failed to create backup: {e}")
                return False
        
        # Restore from snapshot
        snapshot_archive = self.snapshots_dir / f"{snapshot.snapshot_id}.tar.gz"
        
        try:
            # Remove current configuration
            if self.config_dir.exists():
                shutil.rmtree(self.config_dir)
            
            # Extract snapshot
            with tarfile.open(snapshot_archive, 'r:gz') as tar:
                tar.extractall(path=self.config_dir.parent)
            
            self.logger.info(f"Configuration restored from snapshot: {snapshot.snapshot_id}")
            
            if backup_id:
                self.logger.info(f"Pre-rollback backup created: {backup_id}")
            
            return True
            
        except Exception:
            self.logger.error(f"Failed to restore snapshot: {e}")
            
            # Try to restore from backup if available
            if backup_id:
                self.logger.info("Attempting to restore from pre-rollback backup...")
                backup_snapshot = self.get_snapshot(backup_id)
                if backup_snapshot:
                    self.restore_snapshot(backup_snapshot, create_backup=False)
            
            return False
    
    def rollback_with_git(self, snapshot: DeploymentSnapshot) -> bool:
        """Perform rollback using Git operations"""
        self.logger.info(f"Performing Git-based rollback to commit: {snapshot.commit_sha}")
        
        try:
            # Check if we're in a Git repository
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=self.config_dir.parent,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                self.logger.error("Not in a Git repository - cannot perform Git rollback")
                return False
            
            # Checkout the specific commit for config files
            for file_path in snapshot.changed_files:
                if file_path.startswith('config/'):
                    cmd = ["git", "checkout", snapshot.commit_sha, "--", file_path]
                    result = subprocess.run(
                        cmd,
                        cwd=self.config_dir.parent,
                        capture_output=True,
                        text=True
                    )
                    
                    if result.returncode != 0:
                        self.logger.warning(f"Failed to rollback file {file_path}: {result.stderr}")
                    else:
                        self.logger.info(f"Rolled back file: {file_path}")
            
            # Commit the rollback
            commit_message = f"Rollback configuration to snapshot {snapshot.snapshot_id}"
            
            subprocess.run(
                ["git", "add", "config/"],
                cwd=self.config_dir.parent,
                check=True
            )
            
            subprocess.run(
                ["git", "commit", "-m", commit_message],
                cwd=self.config_dir.parent,
                check=True
            )
            
            self.logger.info("Git rollback completed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Git rollback failed: {e}")
            return False
        except Exception:
            self.logger.error(f"Unexpected error during Git rollback: {e}")
            return False
    
    def validate_post_rollback(self, environment: str) -> bool:
        """Validate configuration after rollback"""
        self.logger.info("Validating configuration after rollback...")
        
        # Use the existing validation script
        script_dir = Path(__file__).parent
        validate_script = script_dir / "validate_config_deployment.py"
        
        if not validate_script.exists():
            self.logger.warning("Validation script not found - skipping post-rollback validation")
            return True
        
        try:
            result = subprocess.run([
                sys.executable, str(validate_script),
                "--config-dir", str(self.config_dir),
                "--environment", environment
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                self.logger.info("Post-rollback validation passed")
                return True
            else:
                self.logger.error(f"Post-rollback validation failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("Post-rollback validation timed out")
            return False
        except Exception:
            self.logger.error(f"Failed to run post-rollback validation: {e}")
            return False
    
    def generate_rollback_report(self, snapshot: DeploymentSnapshot, success: bool) -> str:
        """Generate a rollback report"""
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        
        report_lines = [
            "# Configuration Rollback Report",
            f"**Timestamp:** {timestamp}",
            f"**Status:** {'SUCCESS' if success else 'FAILED'}",
            "",
            "## Rollback Details",
            f"- **Target Snapshot:** {snapshot.snapshot_id}",
            f"- **Environment:** {snapshot.environment}",
            f"- **Original Commit:** {snapshot.commit_sha}",
            f"- **Original Author:** {snapshot.author}",
            f"- **Original Message:** {snapshot.commit_message}",
            f"- **Deployment Strategy:** {snapshot.deployment_strategy}",
            "",
            "## Changed Files",
        ]
        
        if snapshot.changed_files:
            for file_path in snapshot.changed_files:
                report_lines.append(f"- {file_path}")
        else:
            report_lines.append("- No specific files tracked")
        
        report_lines.extend([
            "",
            "## Next Steps",
        ])
        
        if success:
            report_lines.extend([
                "- ✅ Configuration has been successfully rolled back",
                "- ✅ Verify that all services are functioning correctly",
                "- ✅ Monitor application metrics for any issues",
                "- ✅ Update documentation if necessary"
            ])
        else:
            report_lines.extend([
                "- ❌ Rollback failed - manual intervention required",
                "- ❌ Check rollback logs for specific error details",
                "- ❌ Consider alternative rollback strategies",
                "- ❌ Escalate to operations team if necessary"
            ])
        
        return "\n".join(report_lines)


def main():
    """Main entry point for the configuration rollback script"""
    parser = argparse.ArgumentParser(
        description="Rollback configuration to a previous deployment snapshot",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("config"),
        help="Path to configuration directory (default: config)"
    )
    
    parser.add_argument(
        "--snapshots-dir",
        type=Path,
        default=Path(".deployment-snapshots"),
        help="Path to deployment snapshots directory (default: .deployment-snapshots)"
    )
    
    parser.add_argument(
        "--environment",
        choices=["development", "staging", "production"],
        help="Filter snapshots by environment"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List snapshots command
    list_parser = subparsers.add_parser("list", help="List available snapshots")
    list_parser.add_argument("--limit", type=int, default=10, help="Limit number of snapshots to show")
    
    # Rollback command
    rollback_parser = subparsers.add_parser("rollback", help="Rollback to a snapshot")
    rollback_parser.add_argument("snapshot_id", help="Snapshot ID to rollback to")
    rollback_parser.add_argument("--no-backup", action="store_true", help="Skip creating pre-rollback backup")
    rollback_parser.add_argument("--use-git", action="store_true", help="Use Git-based rollback")
    rollback_parser.add_argument("--skip-validation", action="store_true", help="Skip post-rollback validation")
    rollback_parser.add_argument("--report-file", type=Path, help="Path to save rollback report")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate a snapshot for rollback")
    validate_parser.add_argument("snapshot_id", help="Snapshot ID to validate")
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    if not args.config_dir.exists():
        print(f"❌ Configuration directory not found: {args.config_dir}")
        sys.exit(1)
    
    # Initialize rollback manager
    rollback_manager = ConfigurationRollback(args.config_dir, args.snapshots_dir)
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == "list":
        snapshots = rollback_manager.list_snapshots(args.environment)
        
        if not snapshots:
            print("No deployment snapshots found")
            sys.exit(0)
        
        print(f"📋 Available deployment snapshots (showing {min(len(snapshots), args.limit)}):")
        print()
        
        for i, snapshot in enumerate(snapshots[:args.limit]):
            print(f"{i+1}. **{snapshot.snapshot_id}**")
            print(f"   Environment: {snapshot.environment}")
            print(f"   Timestamp: {snapshot.timestamp}")
            print(f"   Commit: {snapshot.commit_sha[:8]}")
            print(f"   Author: {snapshot.author}")
            print(f"   Message: {snapshot.commit_message}")
            print(f"   Strategy: {snapshot.deployment_strategy}")
            print(f"   Files: {len(snapshot.changed_files)} changed")
            print()
    
    elif args.command == "validate":
        snapshot = rollback_manager.get_snapshot(args.snapshot_id)
        
        if not snapshot:
            print(f"❌ Snapshot not found: {args.snapshot_id}")
            sys.exit(1)
        
        print(f"🔍 Validating snapshot: {args.snapshot_id}")
        
        is_valid = rollback_manager.validate_rollback_target(snapshot)
        
        if is_valid:
            print(f"✅ Snapshot {args.snapshot_id} is valid for rollback")
            sys.exit(0)
        else:
            print(f"❌ Snapshot {args.snapshot_id} cannot be used for rollback")
            sys.exit(1)
    
    elif args.command == "rollback":
        snapshot = rollback_manager.get_snapshot(args.snapshot_id)
        
        if not snapshot:
            print(f"❌ Snapshot not found: {args.snapshot_id}")
            sys.exit(1)
        
        print(f"🔄 Rolling back to snapshot: {args.snapshot_id}")
        print(f"   Environment: {snapshot.environment}")
        print(f"   Timestamp: {snapshot.timestamp}")
        print(f"   Commit: {snapshot.commit_sha}")
        print()
        
        # Confirm rollback for production
        if snapshot.environment == "production":
            confirm = input("⚠️  This is a PRODUCTION rollback. Type 'CONFIRM' to proceed: ")
            if confirm != "CONFIRM":
                print("Rollback cancelled")
                sys.exit(0)
        
        # Perform rollback
        success = False
        
        if args.use_git:
            success = rollback_manager.rollback_with_git(snapshot)
        else:
            success = rollback_manager.restore_snapshot(snapshot, not args.no_backup)
        
        # Post-rollback validation
        if success and not args.skip_validation:
            validation_success = rollback_manager.validate_post_rollback(snapshot.environment)
            if not validation_success:
                print("⚠️  Rollback completed but validation failed")
                success = False
        
        # Generate report
        report = rollback_manager.generate_rollback_report(snapshot, success)
        
        if args.report_file:
            with open(args.report_file, 'w') as f:
                f.write(report)
            print(f"📊 Rollback report saved to {args.report_file}")
        
        # Final status
        if success:
            print("🎉 Configuration rollback completed successfully!")
            sys.exit(0)
        else:
            print("❌ Configuration rollback failed!")
            sys.exit(1)


if __name__ == "__main__":
    main()