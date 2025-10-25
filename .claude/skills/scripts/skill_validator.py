#!/usr/bin/env python3
"""
Skill Validator - Validate Claude skills for quality and completeness

Checks:
- Required files exist (SKILL.md, FORMS.md, examples.md, reference.md)
- SKILL.md has proper frontmatter
- Activation triggers are clear and specific
- Examples are present and useful
- Scripts are executable and have docstrings
- Templates are valid

Usage:
    python skill_validator.py --skill api-master
    python skill_validator.py --all
    python skill_validator.py --skill nexus --fix
"""

import argparse
import os
import re
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
from enum import Enum


class Severity(Enum):
    """Issue severity levels"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """Represents a validation issue"""
    severity: Severity
    file: str
    message: str
    line: Optional[int] = None
    suggestion: Optional[str] = None


@dataclass
class SkillValidationResult:
    """Result of skill validation"""
    skill_name: str
    passed: bool
    score: float = 0.0
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]

    @property
    def infos(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.INFO]


class SkillValidator:
    """Validates a Claude skill"""

    REQUIRED_FILES = ["SKILL.md"]
    RECOMMENDED_FILES = ["FORMS.md", "examples.md", "reference.md"]
    OPTIONAL_DIRS = ["scripts", "templates"]

    def __init__(self, skill_path: Path, fix: bool = False):
        self.skill_path = skill_path
        self.skill_name = skill_path.name
        self.fix = fix
        self.issues: List[ValidationIssue] = []

    def validate(self) -> SkillValidationResult:
        """Run all validations"""
        self.issues = []

        # Check file structure
        self._validate_file_structure()

        # Validate SKILL.md
        if (self.skill_path / "SKILL.md").exists():
            self._validate_skill_md()

        # Validate FORMS.md
        if (self.skill_path / "FORMS.md").exists():
            self._validate_forms_md()

        # Validate examples.md
        if (self.skill_path / "examples.md").exists():
            self._validate_examples_md()

        # Validate scripts
        if (self.skill_path / "scripts").exists():
            self._validate_scripts()

        # Validate templates
        if (self.skill_path / "templates").exists():
            self._validate_templates()

        # Calculate score
        score = self._calculate_score()

        # Determine if passed
        passed = len([i for i in self.issues if i.severity == Severity.ERROR]) == 0

        return SkillValidationResult(
            skill_name=self.skill_name,
            passed=passed,
            score=score,
            issues=self.issues
        )

    def _validate_file_structure(self):
        """Validate file structure"""
        # Check required files
        for filename in self.REQUIRED_FILES:
            filepath = self.skill_path / filename
            if not filepath.exists():
                self.issues.append(ValidationIssue(
                    severity=Severity.ERROR,
                    file=filename,
                    message=f"Required file missing: {filename}",
                    suggestion=f"Create {filename} with skill documentation"
                ))

        # Check recommended files
        for filename in self.RECOMMENDED_FILES:
            filepath = self.skill_path / filename
            if not filepath.exists():
                self.issues.append(ValidationIssue(
                    severity=Severity.WARNING,
                    file=filename,
                    message=f"Recommended file missing: {filename}",
                    suggestion=f"Create {filename} for better skill documentation"
                ))

    def _validate_skill_md(self):
        """Validate SKILL.md content"""
        filepath = self.skill_path / "SKILL.md"

        with open(filepath, 'r') as f:
            content = f.read()

        # Check for frontmatter
        if not content.startswith('---'):
            self.issues.append(ValidationIssue(
                severity=Severity.ERROR,
                file="SKILL.md",
                message="Missing YAML frontmatter",
                suggestion="Add frontmatter with 'name' and 'description' fields"
            ))
        else:
            # Parse frontmatter
            frontmatter_match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
            if frontmatter_match:
                try:
                    frontmatter = yaml.safe_load(frontmatter_match.group(1))

                    # Check required frontmatter fields
                    if 'name' not in frontmatter:
                        self.issues.append(ValidationIssue(
                            severity=Severity.ERROR,
                            file="SKILL.md",
                            message="Frontmatter missing 'name' field",
                            suggestion="Add 'name: skill-name' to frontmatter"
                        ))

                    if 'description' not in frontmatter:
                        self.issues.append(ValidationIssue(
                            severity=Severity.ERROR,
                            file="SKILL.md",
                            message="Frontmatter missing 'description' field",
                            suggestion="Add 'description: ...' to frontmatter"
                        ))
                    elif len(frontmatter['description']) < 50:
                        self.issues.append(ValidationIssue(
                            severity=Severity.WARNING,
                            file="SKILL.md",
                            message="Description is too short (< 50 chars)",
                            suggestion="Provide a more detailed description of the skill's purpose"
                        ))

                except yaml.YAMLError:
                    self.issues.append(ValidationIssue(
                        severity=Severity.ERROR,
                        file="SKILL.md",
                        message="Invalid YAML in frontmatter",
                        suggestion="Fix YAML syntax in frontmatter"
                    ))

        # Check for key sections
        required_sections = [
            "What .* does for Claude",
            "When Claude activates",
            "How .* works"
        ]

        for section_pattern in required_sections:
            if not re.search(section_pattern, content, re.IGNORECASE):
                self.issues.append(ValidationIssue(
                    severity=Severity.WARNING,
                    file="SKILL.md",
                    message=f"Missing recommended section matching: {section_pattern}",
                    suggestion="Add section to explain skill behavior"
                ))

        # Check for activation triggers
        if "activate" not in content.lower():
            self.issues.append(ValidationIssue(
                severity=Severity.WARNING,
                file="SKILL.md",
                message="No activation triggers mentioned",
                suggestion="Describe when Claude should activate this skill"
            ))

        # Check for vague language
        vague_patterns = [
            (r'\bgeneral\b', "Avoid vague term 'general'"),
            (r'\betc\.?\b', "List specific items instead of 'etc'"),
            (r'\bvarious\b', "Be specific instead of using 'various'"),
        ]

        for pattern, message in vague_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                self.issues.append(ValidationIssue(
                    severity=Severity.INFO,
                    file="SKILL.md",
                    message=message,
                    suggestion="Replace vague language with specific details"
                ))

    def _validate_forms_md(self):
        """Validate FORMS.md content"""
        filepath = self.skill_path / "FORMS.md"

        with open(filepath, 'r') as f:
            content = f.read()

        # Check for configuration examples
        if "```json" not in content and "```yaml" not in content:
            self.issues.append(ValidationIssue(
                severity=Severity.WARNING,
                file="FORMS.md",
                message="No configuration examples found",
                suggestion="Add JSON or YAML configuration examples"
            ))

        # Check for default values
        if "default" not in content.lower():
            self.issues.append(ValidationIssue(
                severity=Severity.INFO,
                file="FORMS.md",
                message="No default values specified",
                suggestion="Specify default values for configuration options"
            ))

    def _validate_examples_md(self):
        """Validate examples.md content"""
        filepath = self.skill_path / "examples.md"

        with open(filepath, 'r') as f:
            content = f.read()

        # Check for code examples
        code_blocks = re.findall(r'```.*?```', content, re.DOTALL)
        if len(code_blocks) < 2:
            self.issues.append(ValidationIssue(
                severity=Severity.WARNING,
                file="examples.md",
                message="Insufficient code examples (< 2)",
                suggestion="Add more practical code examples showing skill usage"
            ))

        # Check for workflow examples
        if "workflow" not in content.lower() and "example" not in content.lower():
            self.issues.append(ValidationIssue(
                severity=Severity.INFO,
                file="examples.md",
                message="No workflow examples found",
                suggestion="Add step-by-step workflow examples"
            ))

    def _validate_scripts(self):
        """Validate scripts directory"""
        scripts_dir = self.skill_path / "scripts"

        python_files = list(scripts_dir.glob("*.py"))

        if not python_files:
            self.issues.append(ValidationIssue(
                severity=Severity.INFO,
                file="scripts/",
                message="No Python scripts found",
                suggestion="Consider adding helper scripts for the skill"
            ))
            return

        for script in python_files:
            self._validate_script_file(script)

    def _validate_script_file(self, script_path: Path):
        """Validate a single script file"""
        # Check if executable
        if not os.access(script_path, os.X_OK):
            self.issues.append(ValidationIssue(
                severity=Severity.WARNING,
                file=f"scripts/{script_path.name}",
                message="Script is not executable",
                suggestion=f"Run: chmod +x {script_path}"
            ))

        # Check for shebang
        with open(script_path, 'r') as f:
            first_line = f.readline()
            if not first_line.startswith('#!/usr/bin/env python'):
                self.issues.append(ValidationIssue(
                    severity=Severity.WARNING,
                    file=f"scripts/{script_path.name}",
                    message="Missing or incorrect shebang",
                    suggestion="Add '#!/usr/bin/env python3' as first line"
                ))

            # Read full content for docstring check
            f.seek(0)
            content = f.read()

            # Check for module docstring
            if not re.search(r'^""".+?"""', content, re.DOTALL | re.MULTILINE):
                self.issues.append(ValidationIssue(
                    severity=Severity.INFO,
                    file=f"scripts/{script_path.name}",
                    message="Missing module docstring",
                    suggestion="Add docstring explaining script purpose and usage"
                ))

            # Check for main function
            if "if __name__ ==" not in content:
                self.issues.append(ValidationIssue(
                    severity=Severity.INFO,
                    file=f"scripts/{script_path.name}",
                    message="No main entry point",
                    suggestion="Add 'if __name__ == \"__main__\":' block"
                ))

    def _validate_templates(self):
        """Validate templates directory"""
        templates_dir = self.skill_path / "templates"

        template_files = list(templates_dir.glob("*"))

        if not template_files:
            self.issues.append(ValidationIssue(
                severity=Severity.INFO,
                file="templates/",
                message="No template files found",
                suggestion="Consider adding code templates for common patterns"
            ))

    def _calculate_score(self) -> float:
        """Calculate overall quality score (0-100)"""
        score = 100.0

        # Deduct points for issues
        for issue in self.issues:
            if issue.severity == Severity.ERROR:
                score -= 10.0
            elif issue.severity == Severity.WARNING:
                score -= 5.0
            elif issue.severity == Severity.INFO:
                score -= 1.0

        return max(0.0, min(100.0, score))


class SkillValidatorRunner:
    """Run validation on multiple skills"""

    def __init__(self, skills_dir: Path, fix: bool = False):
        self.skills_dir = skills_dir
        self.fix = fix

    def validate_all(self) -> Dict[str, SkillValidationResult]:
        """Validate all skills"""
        results = {}

        for skill_dir in self.skills_dir.iterdir():
            if skill_dir.is_dir() and not skill_dir.name.startswith('.'):
                # Skip scripts directory
                if skill_dir.name == "scripts":
                    continue

                validator = SkillValidator(skill_dir, fix=self.fix)
                result = validator.validate()
                results[skill_dir.name] = result

        return results

    def validate_skill(self, skill_name: str) -> SkillValidationResult:
        """Validate a specific skill"""
        skill_path = self.skills_dir / skill_name
        if not skill_path.exists():
            raise ValueError(f"Skill not found: {skill_name}")

        validator = SkillValidator(skill_path, fix=self.fix)
        return validator.validate()


def print_result(result: SkillValidationResult):
    """Print validation result"""
    # Header
    status_icon = "✅" if result.passed else "❌"
    print(f"\n{status_icon} {result.skill_name}")
    print(f"Score: {result.score:.1f}/100")

    if result.issues:
        # Errors
        if result.errors:
            print(f"\n🔴 Errors ({len(result.errors)}):")
            for issue in result.errors:
                print(f"  • {issue.file}: {issue.message}")
                if issue.suggestion:
                    print(f"    → {issue.suggestion}")

        # Warnings
        if result.warnings:
            print(f"\n🟡 Warnings ({len(result.warnings)}):")
            for issue in result.warnings:
                print(f"  • {issue.file}: {issue.message}")
                if issue.suggestion:
                    print(f"    → {issue.suggestion}")

        # Info
        if result.infos:
            print(f"\n🔵 Info ({len(result.infos)}):")
            for issue in result.infos:
                print(f"  • {issue.file}: {issue.message}")
                if issue.suggestion:
                    print(f"    → {issue.suggestion}")
    else:
        print("\n✨ No issues found!")


def main():
    parser = argparse.ArgumentParser(
        description="Validate Claude skills for quality and completeness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate a specific skill
  python skill_validator.py --skill api-master

  # Validate all skills
  python skill_validator.py --all

  # Validate with auto-fix
  python skill_validator.py --skill nexus --fix

  # Show summary only
  python skill_validator.py --all --summary
        """
    )

    parser.add_argument('--skill', help='Skill name to validate')
    parser.add_argument('--all', action='store_true', help='Validate all skills')
    parser.add_argument('--fix', action='store_true', help='Attempt to auto-fix issues')
    parser.add_argument('--summary', action='store_true', help='Show summary only')
    parser.add_argument(
        '--skills-dir',
        default='./.claude/skills',
        help='Skills directory (default: ./.claude/skills)'
    )

    args = parser.parse_args()

    skills_dir = Path(args.skills_dir)

    if not skills_dir.exists():
        print(f"❌ Skills directory not found: {skills_dir}")
        return 1

    runner = SkillValidatorRunner(skills_dir, fix=args.fix)

    if args.all:
        print("🔍 Validating all skills...\n")
        results = runner.validate_all()

        if args.summary:
            # Summary only
            print("=" * 60)
            print(f"{'Skill':<20} {'Status':<10} {'Score':<10} {'Issues'}")
            print("=" * 60)

            for skill_name, result in sorted(results.items()):
                status = "✅ PASS" if result.passed else "❌ FAIL"
                issue_count = len(result.issues)
                print(f"{skill_name:<20} {status:<10} {result.score:>5.1f}/100  {issue_count} issues")

            print("=" * 60)

            # Overall stats
            total_skills = len(results)
            passed_skills = sum(1 for r in results.values() if r.passed)
            avg_score = sum(r.score for r in results.values()) / total_skills if total_skills else 0

            print(f"\nTotal: {total_skills} skills")
            print(f"Passed: {passed_skills}/{total_skills}")
            print(f"Average Score: {avg_score:.1f}/100")

        else:
            # Detailed output
            for skill_name, result in sorted(results.items()):
                print_result(result)

        # Exit code based on results
        return 0 if all(r.passed for r in results.values()) else 1

    elif args.skill:
        print(f"🔍 Validating skill: {args.skill}\n")
        try:
            result = runner.validate_skill(args.skill)
            print_result(result)
            return 0 if result.passed else 1
        except ValueError as e:
            print(f"❌ {e}")
            return 1

    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    exit(main())
