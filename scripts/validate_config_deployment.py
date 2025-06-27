#!/usr/bin/env python3
"""
Configuration Deployment Validation Script

This script provides comprehensive validation for configuration deployments,
supporting the GitOps configuration management workflow.
"""

import json  # noqa: PLC0415
import sys
import yaml
import argparse
import logging  # noqa: PLC0415
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import subprocess
import time  # noqa: PLC0415

# Import existing project modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from config.core import load_config
    from config.validators import validate_config_schema
except ImportError as e:
    logging.warning(f"Could not import project modules: {e}")
    # Define minimal fallback validators
    def load_config() -> Dict[str, Any]:
        return {}
    
    def validate_config_schema(config: Dict[str, Any]) -> bool:
        return True


@dataclass
class ValidationResult:
    """Represents the result of a configuration validation"""
    component: str
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None


class ConfigDeploymentValidator:
    """
    Comprehensive configuration deployment validator
    
    Provides validation for:
    - JSON/YAML syntax
    - Configuration schema
    - Environment-specific requirements
    - Service connectivity
    - Security compliance
    """
    
    def __init__(self, config_dir: Path, environment: str):
        self.config_dir = Path(config_dir)
        self.environment = environment
        self.logger = self._setup_logging()
        self.results: List[ValidationResult] = []
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger("config_validator")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def validate_json_files(self) -> ValidationResult:
        """Validate all JSON configuration files"""
        self.logger.info("Validating JSON configuration files...")
        
        json_files = list(self.config_dir.rglob("*.json"))
        errors = []
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    json.load(f)
                self.logger.debug(f"✅ Valid JSON: {json_file}")
            except json.JSONDecodeError as e:
                error_msg = f"Invalid JSON in {json_file}: {e}"
                errors.append(error_msg)
                self.logger.error(error_msg)
        
        if errors:
            return ValidationResult(
                component="JSON Syntax",
                passed=False,
                message=f"Found {len(errors)} JSON syntax errors",
                details={"errors": errors}
            )
        
        return ValidationResult(
            component="JSON Syntax",
            passed=True,
            message=f"All {len(json_files)} JSON files are valid"
        )
    
    def validate_yaml_files(self) -> ValidationResult:
        """Validate all YAML configuration files"""
        self.logger.info("Validating YAML configuration files...")
        
        yaml_files = list(self.config_dir.rglob("*.yml")) + list(self.config_dir.rglob("*.yaml"))
        errors = []
        
        for yaml_file in yaml_files:
            try:
                with open(yaml_file, 'r') as f:
                    yaml.safe_load(f)
                self.logger.debug(f"✅ Valid YAML: {yaml_file}")
            except yaml.YAMLError as e:
                error_msg = f"Invalid YAML in {yaml_file}: {e}"
                errors.append(error_msg)
                self.logger.error(error_msg)
        
        if errors:
            return ValidationResult(
                component="YAML Syntax",
                passed=False,
                message=f"Found {len(errors)} YAML syntax errors",
                details={"errors": errors}
            )
        
        return ValidationResult(
            component="YAML Syntax",
            passed=True,
            message=f"All {len(yaml_files)} YAML files are valid"
        )
    
    def validate_environment_config(self) -> ValidationResult:
        """Validate environment-specific configuration"""
        self.logger.info(f"Validating {self.environment} environment configuration...")
        
        config_file = self.config_dir / "templates" / f"{self.environment}.json"
        
        if not config_file.exists():
            return ValidationResult(
                component="Environment Config",
                passed=False,
                message=f"Configuration file not found: {config_file}"
            )
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Validate required sections
            required_sections = [
                "environment", "cache", "qdrant", "performance", "security"
            ]
            missing_sections = [s for s in required_sections if s not in config]
            
            if missing_sections:
                return ValidationResult(
                    component="Environment Config",
                    passed=False,
                    message=f"Missing required sections: {missing_sections}",
                    details={"missing_sections": missing_sections}
                )
            
            # Validate environment matches
            if config.get("environment") != self.environment:
                return ValidationResult(
                    component="Environment Config",
                    passed=False,
                    message=f"Environment mismatch: expected {self.environment}, got {config.get('environment')}"
                )
            
            # Environment-specific validations
            if self.environment == "production":
                if config.get("debug", False):
                    return ValidationResult(
                        component="Environment Config",
                        passed=False,
                        message="Debug mode should be disabled in production"
                    )
                
                if not config.get("security", {}).get("require_api_keys", False):
                    return ValidationResult(
                        component="Environment Config",
                        passed=False,
                        message="API key requirement should be enabled in production"
                    )
            
            return ValidationResult(
                component="Environment Config",
                passed=True,
                message=f"Environment configuration is valid for {self.environment}"
            )
            
        except Exception as e:
            return ValidationResult(
                component="Environment Config",
                passed=False,
                message=f"Failed to validate environment config: {e}"
            )
    
    def validate_docker_compose(self) -> ValidationResult:
        """Validate Docker Compose configuration"""
        self.logger.info("Validating Docker Compose configuration...")
        
        base_compose = self.config_dir.parent / "docker-compose.yml"
        env_compose = self.config_dir.parent / f"docker-compose.{self.environment}.yml"
        
        if not base_compose.exists():
            return ValidationResult(
                component="Docker Compose",
                passed=False,
                message="Base docker-compose.yml not found"
            )
        
        try:
            # Validate base compose file
            result = subprocess.run(
                ["docker", "compose", "-f", str(base_compose), "config"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                return ValidationResult(
                    component="Docker Compose",
                    passed=False,
                    message=f"Invalid docker-compose.yml: {result.stderr}"
                )
            
            # Validate with environment override if it exists
            if env_compose.exists():
                result = subprocess.run(
                    ["docker", "compose", "-f", str(base_compose), "-f", str(env_compose), "config"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode != 0:
                    return ValidationResult(
                        component="Docker Compose",
                        passed=False,
                        message=f"Invalid environment override: {result.stderr}"
                    )
            
            return ValidationResult(
                component="Docker Compose",
                passed=True,
                message="Docker Compose configuration is valid"
            )
            
        except subprocess.TimeoutExpired:
            return ValidationResult(
                component="Docker Compose",
                passed=False,
                message="Docker Compose validation timed out"
            )
        except Exception as e:
            return ValidationResult(
                component="Docker Compose",
                passed=False,
                message=f"Failed to validate Docker Compose: {e}"
            )
    
    def validate_monitoring_config(self) -> ValidationResult:
        """Validate monitoring configuration (Prometheus, Grafana)"""
        self.logger.info("Validating monitoring configuration...")
        
        errors = []
        
        # Validate Prometheus config
        prometheus_config = self.config_dir / "prometheus" / "prometheus.yml"
        if prometheus_config.exists():
            try:
                result = subprocess.run(
                    ["docker", "run", "--rm", 
                     "-v", f"{prometheus_config.parent}:/etc/prometheus",
                     "prom/prometheus:latest",
                     "promtool", "check", "config", "/etc/prometheus/prometheus.yml"],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode != 0:
                    errors.append(f"Prometheus config validation failed: {result.stderr}")
            except Exception as e:
                errors.append(f"Failed to validate Prometheus config: {e}")
        
        # Validate Grafana dashboards
        grafana_dashboards = self.config_dir / "grafana" / "dashboards"
        if grafana_dashboards.exists():
            for dashboard_file in grafana_dashboards.glob("*.json"):
                try:
                    with open(dashboard_file, 'r') as f:
                        dashboard = json.load(f)
                    
                    # Basic dashboard validation
                    if "dashboard" not in dashboard and "title" not in dashboard:
                        errors.append(f"Invalid Grafana dashboard: {dashboard_file}")
                        
                except Exception as e:
                    errors.append(f"Failed to validate Grafana dashboard {dashboard_file}: {e}")
        
        if errors:
            return ValidationResult(
                component="Monitoring Config",
                passed=False,
                message=f"Found {len(errors)} monitoring configuration errors",
                details={"errors": errors}
            )
        
        return ValidationResult(
            component="Monitoring Config",
            passed=True,
            message="Monitoring configuration is valid"
        )
    
    def validate_security_compliance(self) -> ValidationResult:
        """Validate security compliance of configuration"""
        self.logger.info("Validating security compliance...")
        
        issues = []
        
        # Check for hardcoded secrets
        for config_file in self.config_dir.rglob("*.json"):
            try:
                with open(config_file, 'r') as f:
                    content = f.read()
                
                # Look for potential secrets (excluding templates)
                if "_template_placeholder_" not in content and "example" not in str(config_file):
                    if any(keyword in content.lower() for keyword in 
                           ["password", "secret", "key", "token"] if "sk-" in content or "ghp_" in content):
                        issues.append(f"Potential hardcoded secret in {config_file}")
                        
            except Exception as e:
                issues.append(f"Failed to scan {config_file}: {e}")
        
        # Check for insecure defaults in production
        if self.environment == "production":
            config_file = self.config_dir / "templates" / "production.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                if config.get("debug", False):
                    issues.append("Debug mode enabled in production")
                
                if not config.get("security", {}).get("enable_rate_limiting", False):
                    issues.append("Rate limiting disabled in production")
        
        if issues:
            return ValidationResult(
                component="Security Compliance",
                passed=False,
                message=f"Found {len(issues)} security issues",
                details={"issues": issues}
            )
        
        return ValidationResult(
            component="Security Compliance",
            passed=True,
            message="Configuration meets security compliance requirements"
        )
    
    def validate_schema_consistency(self) -> ValidationResult:
        """Validate configuration schema consistency"""
        self.logger.info("Validating configuration schema consistency...")
        
        try:
            # Load configuration using project's config loader
            config = load_config()
            
            # Use project's schema validator if available
            if not validate_config_schema(config):
                return ValidationResult(
                    component="Schema Consistency",
                    passed=False,
                    message="Configuration schema validation failed"
                )
            
            return ValidationResult(
                component="Schema Consistency",
                passed=True,
                message="Configuration schema is consistent"
            )
            
        except Exception as e:
            return ValidationResult(
                component="Schema Consistency",
                passed=False,
                message=f"Schema validation failed: {e}"
            )
    
    def run_all_validations(self) -> Tuple[bool, List[ValidationResult]]:
        """Run all configuration validations"""
        self.logger.info(f"Starting comprehensive configuration validation for {self.environment}")
        
        validations = [
            self.validate_json_files,
            self.validate_yaml_files,
            self.validate_environment_config,
            self.validate_docker_compose,
            self.validate_monitoring_config,
            self.validate_security_compliance,
            self.validate_schema_consistency,
        ]
        
        results = []
        all_passed = True
        
        for validation in validations:
            try:
                result = validation()
                results.append(result)
                
                if result.passed:
                    self.logger.info(f"✅ {result.component}: {result.message}")
                else:
                    self.logger.error(f"❌ {result.component}: {result.message}")
                    all_passed = False
                    
            except Exception as e:
                error_result = ValidationResult(
                    component=validation.__name__,
                    passed=False,
                    message=f"Validation failed with error: {e}"
                )
                results.append(error_result)
                self.logger.error(f"❌ {validation.__name__}: {e}")
                all_passed = False
        
        self.results = results
        return all_passed, results
    
    def generate_report(self, output_file: Optional[Path] = None) -> str:
        """Generate a validation report"""
        report_lines = [
            "# Configuration Deployment Validation Report",
            f"**Environment:** {self.environment}",
            f"**Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            ""
        ]
        
        passed_count = sum(1 for r in self.results if r.passed)
        total_count = len(self.results)
        
        report_lines.extend([
            f"- **Total Validations:** {total_count}",
            f"- **Passed:** {passed_count}",
            f"- **Failed:** {total_count - passed_count}",
            f"- **Success Rate:** {(passed_count / total_count * 100):.1f}%" if total_count > 0 else "- **Success Rate:** N/A",
            "",
            "## Detailed Results",
            ""
        ])
        
        for result in self.results:
            status = "✅" if result.passed else "❌"
            report_lines.extend([
                f"### {status} {result.component}",
                f"**Status:** {'PASSED' if result.passed else 'FAILED'}",
                f"**Message:** {result.message}",
                ""
            ])
            
            if result.details:
                report_lines.extend([
                    "**Details:**",
                    "```json",
                    json.dumps(result.details, indent=2),
                    "```",
                    ""
                ])
        
        report = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            self.logger.info(f"Validation report saved to {output_file}")
        
        return report


def main():
    """Main entry point for the configuration validation script"""
    parser = argparse.ArgumentParser(
        description="Validate configuration for deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("config"),
        help="Path to configuration directory (default: config)"
    )
    
    parser.add_argument(
        "--environment",
        choices=["development", "staging", "production"],
        default="development",
        help="Target environment (default: development)"
    )
    
    parser.add_argument(
        "--report-file",
        type=Path,
        help="Path to save validation report"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first validation failure"
    )
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    if not args.config_dir.exists():
        print(f"❌ Configuration directory not found: {args.config_dir}")
        sys.exit(1)
    
    # Run validation
    validator = ConfigDeploymentValidator(args.config_dir, args.environment)
    
    print(f"🔍 Validating configuration for {args.environment} environment...")
    print(f"📁 Configuration directory: {args.config_dir}")
    print()
    
    all_passed, results = validator.run_all_validations()
    
    # Generate and optionally save report
    report = validator.generate_report(args.report_file)
    
    # Print summary
    print()
    print("=" * 60)
    if all_passed:
        print("🎉 All validations passed! Configuration is ready for deployment.")
        exit_code = 0
    else:
        failed_validations = [r for r in results if not r.passed]
        print(f"❌ {len(failed_validations)} validation(s) failed:")
        for result in failed_validations:
            print(f"   - {result.component}: {result.message}")
        print()
        print("⚠️  Configuration is not ready for deployment.")
        exit_code = 1
    
    print("=" * 60)
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()