#!/usr/bin/env python3
"""
Mock Data Detection and Auditing Tool

This tool systematically scans the codebase for instances where mock data
is used instead of real data, which violates scientific accuracy principles.

Critical Detection Patterns:
- Hardcoded fee values (1.5 gwei, etc.)
- Arbitrary constants (Q̄ = 690,000)
- Random number generation in calculations
- Fallback values silently substituted
- Placeholder/mock simulation logic
- Default "optimal" parameter hardcoding

Usage:
    python src/utils/mock_data_detector.py --scan-all
    python src/utils/mock_data_detector.py --file src/optimization/nsga_ii.py
    python src/utils/mock_data_detector.py --check-runtime
"""

import argparse
import os
import re
import sys
import ast
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any
from dataclasses import dataclass
from enum import Enum
import warnings

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class MockDataSeverity(Enum):
    """Severity levels for mock data violations."""
    CRITICAL = "CRITICAL"     # Production code using mock data
    HIGH = "HIGH"             # Fallback logic without warnings
    MEDIUM = "MEDIUM"         # Hardcoded values that should be configurable
    LOW = "LOW"               # Test/development mock data properly labeled
    INFO = "INFO"             # Legitimate use with proper documentation


@dataclass
class MockDataViolation:
    """Represents a detected mock data violation."""
    file_path: str
    line_number: int
    line_content: str
    violation_type: str
    severity: MockDataSeverity
    description: str
    suggested_fix: str
    context_lines: List[str] = None


class MockDataDetector:
    """Comprehensive mock data detection and auditing system."""

    def __init__(self):
        self.violations: List[MockDataViolation] = []
        self.patterns = self._compile_detection_patterns()
        self.whitelist = self._load_whitelist()

    def _compile_detection_patterns(self) -> Dict[str, Dict]:
        """Compile regex patterns for detecting mock data usage."""
        return {
            # Hardcoded fee values
            "hardcoded_fees": {
                "pattern": r"(1\.5\s*\*\s*1e9|1\.5.*gwei|base_fee.*1\.5|minimum.*1\.5)",
                "severity": MockDataSeverity.CRITICAL,
                "description": "Hardcoded 1.5 gwei base fee injection",
                "fix": "Use market-driven fee mechanisms instead of artificial floors"
            },

            # Arbitrary constants
            "arbitrary_constants": {
                "pattern": r"(690[_,]?000|Q[_bar]*\s*=\s*690|q_bar\s*=.*690)",
                "severity": MockDataSeverity.CRITICAL,
                "description": "Arbitrary Q̄ = 690,000 constant",
                "fix": "Replace with empirically measured gas consumption data"
            },

            # Hardcoded gas values
            "hardcoded_gas": {
                "pattern": r"(gasPerTx\s*=\s*200[^0-9]|gas.*=.*200[^0-9]|200\s*gas)",
                "severity": MockDataSeverity.CRITICAL,
                "description": "Hardcoded 200 gas (should be 20,000)",
                "fix": "Use correct gas_per_tx = 20000 from documentation"
            },

            # Random number generation in calculations
            "random_calculations": {
                "pattern": r"(Math\.random\(\).*[\+\-\*\/]|random\(\).*score|np\.random.*fee|\.random\(.*\).*gwei)",
                "severity": MockDataSeverity.HIGH,
                "description": "Random numbers used in fee/score calculations",
                "fix": "Use deterministic calculations based on real data"
            },

            # Fallback logic without warnings
            "silent_fallbacks": {
                "pattern": r"(fallback.*=|getVal\(.*fallback|else.*default|catch.*return)",
                "severity": MockDataSeverity.MEDIUM,
                "description": "Silent fallback values without user notification",
                "fix": "Add explicit warnings when fallback data is used"
            },

            # Mock/placeholder simulation
            "mock_simulation": {
                "pattern": r"(runSimplifiedSimulation|placeholder.*simulation|mock.*simulator|simplified.*scoring)",
                "severity": MockDataSeverity.HIGH,
                "description": "Mock/placeholder simulation logic",
                "fix": "Implement real simulation or fail gracefully with clear error"
            },

            # Optimal parameter hardcoding
            "hardcoded_optimal": {
                "pattern": r"(optimalMu\s*=\s*0\.0|optimalNu\s*=.*|optimal.*=\s*\{.*mu.*0\.0)",
                "severity": MockDataSeverity.MEDIUM,
                "description": "Hardcoded 'optimal' parameters",
                "fix": "Load optimal parameters from configuration or research results"
            },

            # Artificial basefee floors
            "basefee_floors": {
                "pattern": r"(minimum.*1.*gwei|floor.*1e9|min.*basefee.*1\.0|max\(.*1e9)",
                "severity": MockDataSeverity.HIGH,
                "description": "Artificial basefee floor (removed in recent fixes)",
                "fix": "Use realistic floor (0.001 gwei) or remove artificial floors"
            },

            # Mock historical data
            "mock_historical": {
                "pattern": r"(generate.*period|mock.*historical|fake.*data|sample.*data.*=)",
                "severity": MockDataSeverity.MEDIUM,
                "description": "Generated/mock historical data",
                "fix": "Use real historical blockchain data with provenance"
            }
        }

    def _load_whitelist(self) -> Set[str]:
        """Load whitelist of files/patterns that are allowed to use mock data."""
        return {
            # Test files are allowed to use mock data
            "test_",
            "tests/",
            "_test.py",
            ".test.",
            "mock_",
            # Documentation and examples
            "example",
            "demo",
            "README",
            # Specific legitimate cases
            "unit_debug.py",  # Debug tools may use mock data for testing
            "test-unit-safety.html"  # Test interface
        }

    def is_whitelisted(self, file_path: str) -> bool:
        """Check if file is whitelisted for mock data usage."""
        file_path_lower = file_path.lower()
        return any(pattern in file_path_lower for pattern in self.whitelist)

    def scan_file(self, file_path: str) -> List[MockDataViolation]:
        """Scan a single file for mock data violations."""
        violations = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, 1):
                line_stripped = line.strip()
                if not line_stripped or line_stripped.startswith('#'):
                    continue

                for violation_type, pattern_info in self.patterns.items():
                    if re.search(pattern_info["pattern"], line, re.IGNORECASE):
                        # Check if this is a whitelisted file
                        severity = pattern_info["severity"]
                        if self.is_whitelisted(file_path):
                            severity = MockDataSeverity.LOW

                        # Get context lines
                        context_start = max(0, line_num - 3)
                        context_end = min(len(lines), line_num + 2)
                        context = [f"{i+1:4d}: {lines[i].rstrip()}"
                                 for i in range(context_start, context_end)]

                        violation = MockDataViolation(
                            file_path=file_path,
                            line_number=line_num,
                            line_content=line_stripped,
                            violation_type=violation_type,
                            severity=severity,
                            description=pattern_info["description"],
                            suggested_fix=pattern_info["fix"],
                            context_lines=context
                        )
                        violations.append(violation)

        except Exception as e:
            print(f"⚠️  Error scanning {file_path}: {e}")

        return violations

    def scan_directory(self, directory: str, extensions: List[str] = None) -> List[MockDataViolation]:
        """Recursively scan directory for mock data violations."""
        if extensions is None:
            extensions = ['.py', '.js', '.html', '.ts', '.jsx']

        violations = []
        directory_path = Path(directory)

        for ext in extensions:
            for file_path in directory_path.rglob(f"*{ext}"):
                if file_path.is_file():
                    violations.extend(self.scan_file(str(file_path)))

        return violations

    def analyze_violations(self, violations: List[MockDataViolation]) -> Dict[str, Any]:
        """Analyze violations and generate summary statistics."""
        if not violations:
            return {"total": 0, "by_severity": {}, "by_type": {}, "by_file": {}}

        by_severity = {}
        by_type = {}
        by_file = {}

        for violation in violations:
            # Count by severity
            sev = violation.severity.value
            by_severity[sev] = by_severity.get(sev, 0) + 1

            # Count by violation type
            vtype = violation.violation_type
            by_type[vtype] = by_type.get(vtype, 0) + 1

            # Count by file
            file_name = Path(violation.file_path).name
            by_file[file_name] = by_file.get(file_name, 0) + 1

        return {
            "total": len(violations),
            "by_severity": by_severity,
            "by_type": by_type,
            "by_file": by_file,
            "critical_count": by_severity.get("CRITICAL", 0),
            "high_count": by_severity.get("HIGH", 0)
        }

    def generate_report(self, violations: List[MockDataViolation],
                       output_file: str = None) -> str:
        """Generate comprehensive mock data violation report."""
        analysis = self.analyze_violations(violations)

        report = []
        report.append("🔍 MOCK DATA DETECTION REPORT")
        report.append("=" * 60)
        report.append("")

        # Executive Summary
        report.append("📊 EXECUTIVE SUMMARY")
        report.append("-" * 30)
        report.append(f"Total Violations: {analysis['total']}")
        report.append(f"Critical Issues: {analysis['critical_count']}")
        report.append(f"High Priority Issues: {analysis['high_count']}")
        report.append("")

        if analysis["critical_count"] > 0:
            report.append("🚨 CRITICAL: Production code using mock data detected!")
            report.append("   This violates scientific accuracy principles.")
            report.append("")

        # Breakdown by severity
        report.append("📈 VIOLATIONS BY SEVERITY")
        report.append("-" * 30)
        for severity, count in analysis["by_severity"].items():
            icon = "🚨" if severity == "CRITICAL" else "⚠️" if severity == "HIGH" else "ℹ️"
            report.append(f"{icon} {severity}: {count}")
        report.append("")

        # Breakdown by type
        report.append("🏷️  VIOLATIONS BY TYPE")
        report.append("-" * 25)
        for vtype, count in analysis["by_type"].items():
            report.append(f"• {vtype.replace('_', ' ').title()}: {count}")
        report.append("")

        # Most problematic files
        report.append("📁 MOST PROBLEMATIC FILES")
        report.append("-" * 27)
        sorted_files = sorted(analysis["by_file"].items(),
                            key=lambda x: x[1], reverse=True)[:10]
        for file_name, count in sorted_files:
            report.append(f"• {file_name}: {count} violations")
        report.append("")

        # Detailed violations
        if violations:
            report.append("🔎 DETAILED VIOLATIONS")
            report.append("-" * 24)

            # Group by severity for prioritized reporting
            by_severity_grouped = {}
            for violation in violations:
                sev = violation.severity.value
                if sev not in by_severity_grouped:
                    by_severity_grouped[sev] = []
                by_severity_grouped[sev].append(violation)

            # Report critical and high priority first
            priority_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]

            for severity in priority_order:
                if severity not in by_severity_grouped:
                    continue

                sev_violations = by_severity_grouped[severity]
                report.append(f"\n{severity} VIOLATIONS ({len(sev_violations)})")
                report.append("=" * (len(severity) + 12))

                for i, violation in enumerate(sev_violations[:10], 1):  # Limit to 10 per severity
                    report.append(f"\n{i}. {violation.file_path}:{violation.line_number}")
                    report.append(f"   Type: {violation.violation_type}")
                    report.append(f"   Issue: {violation.description}")
                    report.append(f"   Code: {violation.line_content}")
                    report.append(f"   Fix: {violation.suggested_fix}")

                    if violation.context_lines:
                        report.append("   Context:")
                        for context_line in violation.context_lines:
                            prefix = "   >>> " if str(violation.line_number) in context_line[:4] else "       "
                            report.append(f"{prefix}{context_line}")

                if len(sev_violations) > 10:
                    report.append(f"\n   ... and {len(sev_violations) - 10} more {severity} violations")

        # Recommendations
        report.append("\n\n💡 RECOMMENDATIONS")
        report.append("-" * 19)
        report.append("1. **Immediate Action Required** for CRITICAL violations:")
        report.append("   - Remove hardcoded fee values (1.5 gwei base fee)")
        report.append("   - Replace arbitrary constants (Q̄ = 690,000)")
        report.append("   - Fix incorrect gas values (200 → 20,000)")
        report.append("")
        report.append("2. **High Priority** for production deployment:")
        report.append("   - Remove mock simulation fallback logic")
        report.append("   - Add warnings for all fallback data usage")
        report.append("   - Implement real data validation")
        report.append("")
        report.append("3. **Implementation Guidelines**:")
        report.append("   - Label ALL mock data as 'MOCK/TEST/FALLBACK'")
        report.append("   - Add runtime warnings for non-real data")
        report.append("   - Prefer failure over silent mock substitution")
        report.append("   - Implement data provenance tracking")
        report.append("")

        report_text = "\n".join(report)

        if output_file:
            try:
                with open(output_file, 'w') as f:
                    f.write(report_text)
                print(f"📄 Report saved to: {output_file}")
            except Exception as e:
                print(f"⚠️  Failed to save report: {e}")

        return report_text

    def check_runtime_mock_usage(self) -> Dict[str, Any]:
        """Check for runtime mock data usage in key system components."""
        runtime_checks = {
            "canonical_modules": self._check_canonical_modules(),
            "optimization_system": self._check_optimization_system(),
            "simulation_engine": self._check_simulation_engine(),
            "data_loading": self._check_data_loading()
        }

        return runtime_checks

    def _check_canonical_modules(self) -> Dict[str, str]:
        """Check if canonical modules use any mock data."""
        results = {}
        canonical_files = [
            "src/core/canonical_fee_mechanism.py",
            "canonical-fee-mechanism.js"
        ]

        for file_path in canonical_files:
            if os.path.exists(file_path):
                violations = self.scan_file(file_path)
                critical_violations = [v for v in violations if v.severity == MockDataSeverity.CRITICAL]
                if critical_violations:
                    results[file_path] = f"❌ {len(critical_violations)} critical mock data violations"
                else:
                    results[file_path] = "✅ No critical mock data usage detected"
            else:
                results[file_path] = "⚠️  File not found"

        return results

    def _check_optimization_system(self) -> Dict[str, str]:
        """Check optimization system for mock data usage."""
        results = {}
        opt_files = [
            "nsga-ii-web.js",
            "app.js",
            "src/optimization/"
        ]

        for file_path in opt_files:
            if os.path.exists(file_path):
                if os.path.isfile(file_path):
                    violations = self.scan_file(file_path)
                else:
                    violations = self.scan_directory(file_path, ['.py'])

                mock_violations = [v for v in violations if "mock" in v.violation_type or "random" in v.violation_type]
                if mock_violations:
                    results[file_path] = f"❌ {len(mock_violations)} mock calculation violations"
                else:
                    results[file_path] = "✅ Real optimization detected"
            else:
                results[file_path] = "⚠️  Path not found"

        return results

    def _check_simulation_engine(self) -> Dict[str, str]:
        """Check simulation components for mock data."""
        results = {}
        sim_files = [
            "simulator.js",
            "specs-simulator.js",
            "src/simulation/"
        ]

        for file_path in sim_files:
            if os.path.exists(file_path):
                if os.path.isfile(file_path):
                    violations = self.scan_file(file_path)
                else:
                    violations = self.scan_directory(file_path, ['.py'])

                fallback_violations = [v for v in violations if "fallback" in v.violation_type]
                if fallback_violations:
                    results[file_path] = f"⚠️  {len(fallback_violations)} silent fallback patterns"
                else:
                    results[file_path] = "✅ Real simulation data usage"
            else:
                results[file_path] = "⚠️  Path not found"

        return results

    def _check_data_loading(self) -> Dict[str, str]:
        """Check data loading systems for mock data injection."""
        results = {}
        data_files = [
            "src/data/",
            "src/utils/quick_alpha_validation.py"
        ]

        for file_path in data_files:
            if os.path.exists(file_path):
                if os.path.isfile(file_path):
                    violations = self.scan_file(file_path)
                else:
                    violations = self.scan_directory(file_path, ['.py'])

                hardcode_violations = [v for v in violations
                                     if v.severity in [MockDataSeverity.CRITICAL, MockDataSeverity.HIGH]]
                if hardcode_violations:
                    results[file_path] = f"❌ {len(hardcode_violations)} hardcoded data violations"
                else:
                    results[file_path] = "✅ Real data loading detected"
            else:
                results[file_path] = "⚠️  Path not found"

        return results

    def create_enforcement_config(self) -> Dict[str, Any]:
        """Create configuration for mock data enforcement system."""
        return {
            "enforcement_rules": {
                "production_paths": [
                    "src/core/",
                    "src/optimization/",
                    "canonical-*.js",
                    "*.js"  # Main web interface files
                ],
                "allowed_mock_paths": [
                    "test*/",
                    "src/tests/",
                    "*test*.py",
                    "*test*.js",
                    "src/utils/unit_debug.py",
                    "docs/",
                    "examples/"
                ],
                "banned_patterns": [
                    r"1\.5\s*\*\s*1e9",  # 1.5 gwei hardcode
                    r"690[_,]?000",      # Arbitrary Q̄ constant
                    r"gasPerTx\s*=\s*200[^0-9]",  # Wrong gas value
                    r"Math\.random\(\).*score",    # Random in calculations
                    r"runSimplifiedSimulation",    # Mock simulation
                ],
                "required_warnings": [
                    "fallback",
                    "default",
                    "mock",
                    "placeholder"
                ]
            },
            "runtime_checks": {
                "validate_data_sources": True,
                "warn_on_fallbacks": True,
                "fail_on_mock_in_production": True,
                "log_data_provenance": True
            },
            "ci_integration": {
                "fail_build_on_critical": True,
                "max_allowed_high_violations": 0,
                "generate_report_artifact": True
            }
        }


def main():
    """Main CLI interface for mock data detection."""
    parser = argparse.ArgumentParser(description="Mock Data Detection and Auditing Tool")

    parser.add_argument("--scan-all", action="store_true",
                       help="Scan entire codebase for mock data violations")
    parser.add_argument("--file", type=str,
                       help="Scan specific file for violations")
    parser.add_argument("--directory", type=str,
                       help="Scan specific directory for violations")
    parser.add_argument("--check-runtime", action="store_true",
                       help="Check runtime mock data usage in key components")
    parser.add_argument("--output", type=str,
                       help="Output report to file")
    parser.add_argument("--severity", type=str, choices=["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"],
                       help="Filter by minimum severity level")
    parser.add_argument("--create-config", action="store_true",
                       help="Create enforcement configuration file")
    parser.add_argument("--json", action="store_true",
                       help="Output results in JSON format")

    args = parser.parse_args()

    detector = MockDataDetector()
    violations = []

    if args.scan_all:
        print("🔍 Scanning entire codebase for mock data violations...")
        violations = detector.scan_directory(".")

    elif args.file:
        if os.path.exists(args.file):
            print(f"🔍 Scanning {args.file}...")
            violations = detector.scan_file(args.file)
        else:
            print(f"❌ File not found: {args.file}")
            return 1

    elif args.directory:
        if os.path.exists(args.directory):
            print(f"🔍 Scanning directory {args.directory}...")
            violations = detector.scan_directory(args.directory)
        else:
            print(f"❌ Directory not found: {args.directory}")
            return 1

    elif args.check_runtime:
        print("🔍 Checking runtime mock data usage...")
        runtime_results = detector.check_runtime_mock_usage()

        print("\n📊 RUNTIME MOCK DATA CHECK RESULTS")
        print("=" * 40)

        for component, checks in runtime_results.items():
            print(f"\n{component.replace('_', ' ').title()}:")
            for file_path, status in checks.items():
                print(f"  {status} {file_path}")

        return 0

    elif args.create_config:
        config = detector.create_enforcement_config()
        config_file = "mock_data_enforcement.json"

        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"📄 Mock data enforcement config created: {config_file}")
        return 0

    else:
        print("❌ No action specified. Use --scan-all, --file, --directory, or --check-runtime")
        return 1

    # Filter by severity if specified
    if args.severity:
        target_severity = MockDataSeverity(args.severity)
        severity_levels = {
            MockDataSeverity.CRITICAL: 0,
            MockDataSeverity.HIGH: 1,
            MockDataSeverity.MEDIUM: 2,
            MockDataSeverity.LOW: 3,
            MockDataSeverity.INFO: 4
        }
        min_level = severity_levels[target_severity]
        violations = [v for v in violations
                     if severity_levels[v.severity] <= min_level]

    if args.json:
        # Output JSON format
        violation_data = []
        for v in violations:
            violation_data.append({
                "file": v.file_path,
                "line": v.line_number,
                "type": v.violation_type,
                "severity": v.severity.value,
                "description": v.description,
                "code": v.line_content,
                "fix": v.suggested_fix
            })

        result = {
            "summary": detector.analyze_violations(violations),
            "violations": violation_data
        }

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
        else:
            print(json.dumps(result, indent=2))
    else:
        # Generate and display report
        report = detector.generate_report(violations, args.output)
        if not args.output:
            print(report)

    # Return appropriate exit code
    analysis = detector.analyze_violations(violations)
    if analysis["critical_count"] > 0:
        return 2  # Critical issues found
    elif analysis["high_count"] > 0:
        return 1  # High priority issues found
    else:
        return 0  # No serious issues


if __name__ == "__main__":
    sys.exit(main())