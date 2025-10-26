#!/usr/bin/env python3
"""
PRD-TASKMASTER Analyzer
Analyzes PRD documents and task lists to identify skill needs.

Workflow:
1. Find and parse PRD/task files
2. Extract tasks and requirements
3. Cluster tasks by domain/pattern
4. Identify skill needs
5. Generate NEXUS directives
"""

import re
import json
import os
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
import argparse


class TaskPattern:
    """Represents a detected task pattern"""
    def __init__(self, pattern_type: str, tasks: List[str], frequency: int):
        self.pattern_type = pattern_type
        self.tasks = tasks
        self.frequency = frequency


class PRDAnalyzer:
    """Analyzes PRD and task documents for skill needs"""

    # Domain keyword patterns
    DOMAIN_PATTERNS = {
        "api": ["api", "endpoint", "rest", "graphql", "request", "response", "http", "webhook"],
        "testing": ["test", "testing", "unit test", "integration test", "e2e", "coverage", "assert"],
        "deployment": ["deploy", "docker", "kubernetes", "k8s", "ci/cd", "pipeline", "container", "helm"],
        "documentation": ["readme", "docs", "documentation", "wiki", "guide", "tutorial", "comment"],
        "database": ["database", "sql", "query", "migration", "schema", "postgres", "mongo", "orm"],
        "frontend": ["ui", "frontend", "react", "vue", "angular", "component", "css", "html"],
        "backend": ["backend", "server", "api", "service", "microservice", "handler"],
        "performance": ["performance", "optimize", "cache", "speed", "latency", "throughput", "profiling"],
        "security": ["security", "auth", "authentication", "authorization", "encrypt", "permission", "jwt"],
        "data_processing": ["data", "etl", "transform", "parse", "csv", "json", "xml", "process"]
    }

    # Skill mappings
    SKILL_RECOMMENDATIONS = {
        "api": {"name": "api-master", "priority": "high"},
        "testing": {"name": "test-guardian", "priority": "high"},
        "deployment": {"name": "deploy-sage", "priority": "medium"},
        "documentation": {"name": "doc-genius", "priority": "low"},
        "database": {"name": "db-wizard", "priority": "medium"},
        "performance": {"name": "perf-optimizer", "priority": "medium"},
        "security": {"name": "security-shield", "priority": "high"},
        "data_processing": {"name": "data-wizard", "priority": "medium"}
    }

    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.prd_files = []
        self.tasks = []
        self.task_patterns = defaultdict(list)

    def find_prd_files(self) -> List[Path]:
        """Find PRD and task files in repository"""
        patterns = [
            "*[Pp][Rr][Dd]*.md",
            "*[Tt][Oo][Dd][Oo]*.md",
            "*[Tt][Aa][Ss][Kk]*.md",
            "*[Rr][Ee][Qq][Uu][Ii][Rr][Ee][Mm][Ee][Nn][Tt]*.md",
            "*[Rr][Oo][Aa][Dd][Mm][Aa][Pp]*.md"
        ]

        found_files = []
        for pattern in patterns:
            found_files.extend(self.repo_path.glob(pattern))

        self.prd_files = list(set(found_files))  # Remove duplicates
        return self.prd_files

    def parse_tasks_from_file(self, filepath: Path) -> List[Dict]:
        """Extract tasks from markdown file"""
        tasks = []

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # Find task patterns
            # Pattern 1: - [ ] Task or - [x] Task
            checkbox_tasks = re.findall(r'^[-*]\s*\[[ xX]\]\s*(.+)$', content, re.MULTILINE)
            tasks.extend([{"text": t.strip(), "file": str(filepath), "type": "checkbox"} for t in checkbox_tasks])

            # Pattern 2: Numbered lists (1. Task)
            numbered_tasks = re.findall(r'^\d+\.\s+(.+)$', content, re.MULTILINE)
            # Filter out short lines (likely headers)
            numbered_tasks = [t for t in numbered_tasks if len(t) > 15]
            tasks.extend([{"text": t.strip(), "file": str(filepath), "type": "numbered"} for t in numbered_tasks])

            # Pattern 3: Bullet points (- Task or * Task)
            bullet_tasks = re.findall(r'^[-*]\s+([^[].+)$', content, re.MULTILINE)
            # Filter duplicates with checkbox tasks and short lines
            bullet_tasks = [t for t in bullet_tasks if len(t) > 15 and not t.strip().startswith('[')]
            tasks.extend([{"text": t.strip(), "file": str(filepath), "type": "bullet"} for t in bullet_tasks])

            # Pattern 4: Headers that look like tasks (### Task name)
            header_tasks = re.findall(r'^#{3,4}\s+(.+)$', content, re.MULTILINE)
            header_tasks = [t for t in header_tasks if any(keyword in t.lower() for keyword in ['implement', 'create', 'build', 'add', 'fix'])]
            tasks.extend([{"text": t.strip(), "file": str(filepath), "type": "header"} for t in header_tasks])

        except Exception as e:
            print(f"Error parsing {filepath}: {e}")

        return tasks

    def parse_all_files(self) -> List[Dict]:
        """Parse all found PRD/task files"""
        all_tasks = []

        for filepath in self.prd_files:
            print(f"Parsing: {filepath}")
            file_tasks = self.parse_tasks_from_file(filepath)
            all_tasks.extend(file_tasks)
            print(f"  Found {len(file_tasks)} tasks")

        self.tasks = all_tasks
        return all_tasks

    def classify_task(self, task_text: str) -> List[str]:
        """Classify a task into domain categories"""
        task_lower = task_text.lower()
        domains = []

        for domain, keywords in self.DOMAIN_PATTERNS.items():
            if any(keyword in task_lower for keyword in keywords):
                domains.append(domain)

        return domains if domains else ["general"]

    def analyze_task_patterns(self) -> Dict[str, TaskPattern]:
        """Analyze tasks and group by patterns"""
        domain_tasks = defaultdict(list)

        for task in self.tasks:
            task_text = task["text"]
            domains = self.classify_task(task_text)

            for domain in domains:
                domain_tasks[domain].append(task_text)

        # Create TaskPattern objects
        patterns = {}
        for domain, tasks in domain_tasks.items():
            if len(tasks) >= 2:  # Only create pattern if 2+ tasks
                patterns[domain] = TaskPattern(
                    pattern_type=domain,
                    tasks=tasks,
                    frequency=len(tasks)
                )

        self.task_patterns = patterns
        return patterns

    def generate_skill_recommendations(self) -> List[Dict]:
        """Generate skill recommendations based on patterns"""
        recommendations = []

        for domain, pattern in self.task_patterns.items():
            skill_info = self.SKILL_RECOMMENDATIONS.get(domain)

            if not skill_info:
                continue

            # Calculate priority based on frequency
            base_priority = skill_info["priority"]
            if pattern.frequency >= 8:
                priority = "critical"
            elif pattern.frequency >= 5:
                priority = "high"
            elif pattern.frequency >= 3:
                priority = "medium"
            else:
                priority = "low"

            recommendations.append({
                "domain": domain,
                "skill_name": skill_info["name"],
                "priority": priority,
                "task_count": pattern.frequency,
                "example_tasks": pattern.tasks[:5],
                "reason": f"Detected {pattern.frequency} tasks related to {domain}",
                "recommended_capabilities": self._get_capabilities_for_domain(domain)
            })

        # Sort by task count (descending)
        recommendations.sort(key=lambda x: x["task_count"], reverse=True)

        return recommendations

    def _get_capabilities_for_domain(self, domain: str) -> List[str]:
        """Get recommended capabilities for a domain"""
        capabilities = {
            "api": [
                "Rate limiting and retry logic",
                "Error handling patterns",
                "Response caching",
                "API client generation"
            ],
            "testing": [
                "Test case generation",
                "Coverage analysis",
                "Test fixture creation",
                "Assertion helpers"
            ],
            "deployment": [
                "Docker optimization",
                "CI/CD pipeline generation",
                "Environment management",
                "Deployment automation"
            ],
            "documentation": [
                "README generation",
                "API documentation",
                "Code commenting",
                "Usage examples"
            ],
            "database": [
                "Query optimization",
                "Migration generation",
                "Schema validation",
                "ORM helpers"
            ],
            "performance": [
                "Bottleneck detection",
                "Caching strategies",
                "Profiling tools",
                "Optimization suggestions"
            ],
            "security": [
                "Vulnerability scanning",
                "Authentication helpers",
                "Input validation",
                "Security best practices"
            ],
            "data_processing": [
                "ETL pipeline automation",
                "Data validation",
                "Format conversion",
                "Transformation helpers"
            ]
        }

        return capabilities.get(domain, [])

    def generate_analysis_report(self) -> Dict:
        """Generate comprehensive analysis report"""
        recommendations = self.generate_skill_recommendations()

        report = {
            "analysis_timestamp": Path(__file__).stat().st_mtime,
            "repository": str(self.repo_path),
            "prd_files_analyzed": [str(f) for f in self.prd_files],
            "total_tasks": len(self.tasks),
            "summary": {
                "total_patterns_detected": len(self.task_patterns),
                "skills_recommended": len(recommendations),
                "high_priority_skills": len([r for r in recommendations if r["priority"] in ["high", "critical"]]),
                "task_coverage": sum(r["task_count"] for r in recommendations)
            },
            "task_patterns": {
                domain: {
                    "frequency": pattern.frequency,
                    "example_tasks": pattern.tasks[:3]
                }
                for domain, pattern in self.task_patterns.items()
            },
            "skill_recommendations": recommendations
        }

        return report

    def save_report(self, report: Dict, output_path: str):
        """Save analysis report to JSON"""
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {output_path}")

    def print_summary(self, report: Dict):
        """Print analysis summary to console"""
        print("\n" + "=" * 80)
        print("PRD-TASKMASTER ANALYSIS SUMMARY")
        print("=" * 80)

        summary = report["summary"]
        print(f"\nTotal Tasks Analyzed: {report['total_tasks']}")
        print(f"Patterns Detected: {summary['total_patterns_detected']}")
        print(f"Skills Recommended: {summary['skills_recommended']}")
        print(f"High Priority Skills: {summary['high_priority_skills']}")

        print("\n" + "-" * 80)
        print("RECOMMENDED SKILLS (by priority):")
        print("-" * 80)

        for rec in report["skill_recommendations"]:
            priority_emoji = {
                "critical": "🔴",
                "high": "🟠",
                "medium": "🟡",
                "low": "🟢"
            }.get(rec["priority"], "⚪")

            print(f"\n{priority_emoji} {rec['skill_name'].upper()} ({rec['priority'].upper()})")
            print(f"   Domain: {rec['domain']}")
            print(f"   Tasks: {rec['task_count']}")
            print(f"   Reason: {rec['reason']}")
            print(f"   Example tasks:")
            for i, task in enumerate(rec['example_tasks'][:3], 1):
                task_preview = task[:70] + "..." if len(task) > 70 else task
                print(f"     {i}. {task_preview}")

        print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="PRD-TASKMASTER - Analyze PRDs and task lists for skill needs"
    )

    parser.add_argument(
        "--repo",
        help="Repository path (default: current directory)",
        type=str,
        default="."
    )

    parser.add_argument(
        "--output",
        help="Output path for analysis report",
        type=str,
        default="prd_analysis_report.json"
    )

    parser.add_argument(
        "--quiet",
        help="Suppress console output",
        action="store_true"
    )

    args = parser.parse_args()

    # Run analysis
    analyzer = PRDAnalyzer(args.repo)

    print(f"Searching for PRD and task files in: {args.repo}")
    prd_files = analyzer.find_prd_files()

    if not prd_files:
        print("\n⚠️  No PRD or task files found")
        print("Looking for: *PRD*.md, *TODO*.md, *TASK*.md, *REQUIREMENTS*.md")
        return 1

    print(f"\nFound {len(prd_files)} file(s):")
    for f in prd_files:
        print(f"  - {f}")

    print("\n" + "-" * 80)
    analyzer.parse_all_files()

    print("\n" + "-" * 80)
    print("Analyzing task patterns...")
    analyzer.analyze_task_patterns()

    report = analyzer.generate_analysis_report()

    # Save report
    analyzer.save_report(report, args.output)

    # Print summary
    if not args.quiet:
        analyzer.print_summary(report)

    print(f"\n✓ Analysis complete!")
    print(f"Next step: Use NEXUS directive generator with this report to create skill directives")

    return 0


if __name__ == "__main__":
    exit(main())
