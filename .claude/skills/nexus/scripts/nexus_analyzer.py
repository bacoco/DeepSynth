#!/usr/bin/env python3
"""
NEXUS Unified Analyzer
Analyzes SOUL memory + PRD + tasks + code to recommend skills for generation.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

# Add SOUL to path
soul_scripts_path = Path(__file__).parent.parent.parent / "soul" / "scripts"
sys.path.insert(0, str(soul_scripts_path))

try:
    from soul_api import get_pattern_analysis, get_soul_memory, get_current_context
    SOUL_AVAILABLE = True
except ImportError:
    SOUL_AVAILABLE = False
    print("⚠️  SOUL not available. Limited analysis mode.")

# Import PRD analyzer
try:
    from prd_analyzer import PRDAnalyzer
    PRD_ANALYZER_AVAILABLE = True
except ImportError:
    PRD_ANALYZER_AVAILABLE = False
    print("⚠️  PRD Analyzer not available.")


class NEXUSUnifiedAnalyzer:
    """
    Unified analyzer that combines:
    - SOUL memory patterns
    - PRD/requirements analysis
    - TODO/task analysis
    - Existing code analysis (optional)
    """

    def __init__(self, repo_path: str = ".", threshold: int = 5, days: int = 7):
        self.repo_path = Path(repo_path)
        self.threshold = threshold
        self.days = days
        self.recommendations = []

    def analyze_soul_patterns(self) -> List[Dict]:
        """Analyze SOUL memory for recurring patterns"""
        if not SOUL_AVAILABLE:
            return []

        try:
            analysis = get_pattern_analysis(days=self.days, threshold=self.threshold)
            recommendations = []

            for pattern_type, pattern_info in analysis.get("patterns", {}).items():
                rec = {
                    "source": "SOUL patterns",
                    "pattern_type": pattern_type,
                    "skill_name": pattern_info.get("suggested_skill", f"{pattern_type}-handler"),
                    "priority": pattern_info.get("priority", "low"),
                    "frequency": pattern_info.get("frequency", 0),
                    "count": pattern_info.get("count", 0),
                    "reason": f"Detected {pattern_info.get('count', 0)} occurrences in {self.days} days",
                    "contexts": pattern_info.get("contexts", [])[:3]
                }
                recommendations.append(rec)

            return recommendations

        except Exception as e:
            print(f"Error analyzing SOUL patterns: {e}")
            return []

    def analyze_prd_and_tasks(self) -> List[Dict]:
        """Analyze PRD files and task lists"""
        if not PRD_ANALYZER_AVAILABLE:
            return []

        try:
            analyzer = PRDAnalyzer(str(self.repo_path))

            # Find and parse PRD files
            prd_files = analyzer.find_prd_files()
            if not prd_files:
                return []

            analyzer.parse_all_files()
            analyzer.analyze_task_patterns()

            # Get skill recommendations
            recommendations = analyzer.generate_skill_recommendations()

            # Format for unified output
            for rec in recommendations:
                rec["source"] = "PRD/Tasks analysis"

            return recommendations

        except Exception as e:
            print(f"Error analyzing PRD: {e}")
            return []

    def analyze_existing_code(self) -> List[Dict]:
        """
        Analyze existing codebase structure
        (Optional, for future enhancement)
        """
        # TODO: Implement code analysis
        # Could detect:
        # - Existing patterns in code (API calls, DB queries, etc.)
        # - Missing test coverage
        # - Documentation gaps
        return []

    def get_existing_skills(self) -> List[str]:
        """Get list of existing skills to avoid duplicates"""
        skills_dir = self.repo_path / ".claude" / "skills"
        if not skills_dir.exists():
            return []

        existing = []
        for item in skills_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                existing.append(item.name)

        return existing

    def merge_and_prioritize(self, all_recommendations: List[Dict]) -> List[Dict]:
        """
        Merge recommendations from different sources and prioritize
        Remove duplicates and existing skills
        """
        existing_skills = self.get_existing_skills()

        # Remove recommendations for existing skills
        filtered = [
            rec for rec in all_recommendations
            if rec["skill_name"] not in existing_skills
        ]

        # Group by skill name and merge
        merged = {}
        for rec in filtered:
            skill_name = rec["skill_name"]
            if skill_name not in merged:
                merged[skill_name] = rec
            else:
                # Merge: take highest priority
                current = merged[skill_name]
                priority_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}

                if priority_order.get(rec["priority"], 0) > priority_order.get(current["priority"], 0):
                    merged[skill_name] = rec

                # Combine sources
                sources = current.get("source", "") + " + " + rec.get("source", "")
                merged[skill_name]["source"] = sources

        # Convert back to list and sort by priority
        result = list(merged.values())
        priority_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        result.sort(key=lambda x: priority_order.get(x["priority"], 0), reverse=True)

        return result

    def analyze_all(self) -> List[Dict]:
        """Run complete analysis from all sources"""
        print(f"🔍 NEXUS Unified Analysis (threshold: {self.threshold}, days: {self.days})")
        print(f"   Repository: {self.repo_path}\n")

        all_recommendations = []

        # 1. SOUL patterns
        print("📊 Analyzing SOUL memory patterns...")
        soul_recs = self.analyze_soul_patterns()
        print(f"   Found {len(soul_recs)} pattern-based recommendations\n")
        all_recommendations.extend(soul_recs)

        # 2. PRD/Tasks
        print("📋 Analyzing PRD and task files...")
        prd_recs = self.analyze_prd_and_tasks()
        print(f"   Found {len(prd_recs)} PRD-based recommendations\n")
        all_recommendations.extend(prd_recs)

        # 3. Code analysis (future)
        # code_recs = self.analyze_existing_code()
        # all_recommendations.extend(code_recs)

        # Merge and prioritize
        print("🔄 Merging and prioritizing recommendations...")
        final_recommendations = self.merge_and_prioritize(all_recommendations)
        print(f"   Final: {len(final_recommendations)} unique skill recommendations\n")

        self.recommendations = final_recommendations
        return final_recommendations

    def generate_recommendations_file(self, output_path: str = "NEXUS_RECOMMENDATIONS.md"):
        """Generate markdown file with skill recommendations for Claude to read"""

        output_file = self.repo_path / output_path

        content = f"""# NEXUS Skill Recommendations

> Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
> Analysis period: Last {self.days} days
> Pattern threshold: {self.threshold} occurrences

---

## Summary

- **Total recommendations**: {len(self.recommendations)}
- **High priority**: {len([r for r in self.recommendations if r['priority'] in ['critical', 'high']])}
- **Medium priority**: {len([r for r in self.recommendations if r['priority'] == 'medium'])}
- **Low priority**: {len([r for r in self.recommendations if r['priority'] == 'low'])}

---

## Recommended Skills

"""

        if not self.recommendations:
            content += "\n✅ **No new skills needed** - existing skills cover current patterns.\n"
        else:
            for i, rec in enumerate(self.recommendations, 1):
                priority_emoji = {
                    "critical": "🔴",
                    "high": "🟠",
                    "medium": "🟡",
                    "low": "🟢"
                }.get(rec["priority"], "⚪")

                content += f"\n### {i}. {priority_emoji} {rec['skill_name']}\n\n"
                content += f"**Priority:** {rec['priority'].upper()}\n\n"
                content += f"**Pattern Type:** {rec.get('pattern_type', 'N/A')}\n\n"
                content += f"**Source:** {rec['source']}\n\n"
                content += f"**Reason:** {rec['reason']}\n\n"

                # Add frequency if available
                if 'frequency' in rec and rec['frequency'] > 0:
                    content += f"**Frequency:** {rec['frequency']:.1f} times/day ({rec.get('count', 0)} total)\n\n"

                # Add task count if from PRD
                if 'task_count' in rec:
                    content += f"**Related Tasks:** {rec['task_count']}\n\n"

                # Add capabilities if available
                if 'recommended_capabilities' in rec and rec['recommended_capabilities']:
                    content += "**Recommended Capabilities:**\n"
                    for cap in rec['recommended_capabilities'][:5]:
                        content += f"- {cap}\n"
                    content += "\n"

                # Add example contexts
                if 'contexts' in rec and rec['contexts']:
                    content += "**Example Contexts:**\n"
                    for ctx in rec['contexts'][:3]:
                        if isinstance(ctx, dict):
                            desc = ctx.get('description', str(ctx))
                            content += f"- {desc}\n"
                        else:
                            content += f"- {ctx}\n"
                    content += "\n"

                # Add example tasks if from PRD
                if 'example_tasks' in rec and rec['example_tasks']:
                    content += "**Example Tasks:**\n"
                    for task in rec['example_tasks'][:3]:
                        task_preview = task[:80] + "..." if len(task) > 80 else task
                        content += f"- {task_preview}\n"
                    content += "\n"

                content += "---\n"

        content += f"""
## Next Steps

1. **Review** these recommendations
2. **Use the skill-generator meta-skill** to create high-priority skills
3. **Test** generated skills with your workflow
4. **NEXUS will continue monitoring** and update recommendations

---

*Generated by NEXUS Unified Analyzer*
*Combining SOUL patterns, PRD analysis, and task analysis*
"""

        with open(output_file, 'w') as f:
            f.write(content)

        print(f"✅ Recommendations saved to: {output_file}")
        return output_file


def main():
    import argparse

    parser = argparse.ArgumentParser(description="NEXUS Unified Analyzer")
    parser.add_argument("--repo", default=".", help="Repository path")
    parser.add_argument("--threshold", type=int, default=5, help="Pattern threshold")
    parser.add_argument("--days", type=int, default=7, help="Days to analyze")
    parser.add_argument("--output", default="NEXUS_RECOMMENDATIONS.md", help="Output file")

    args = parser.parse_args()

    # Run analysis
    analyzer = NEXUSUnifiedAnalyzer(
        repo_path=args.repo,
        threshold=args.threshold,
        days=args.days
    )

    recommendations = analyzer.analyze_all()

    # Generate recommendations file
    analyzer.generate_recommendations_file(args.output)

    # Print summary
    print("\n" + "="*80)
    print("NEXUS ANALYSIS COMPLETE")
    print("="*80)

    if recommendations:
        print(f"\n🎯 Top recommendation: {recommendations[0]['skill_name']}")
        print(f"   Priority: {recommendations[0]['priority']}")
        print(f"   Source: {recommendations[0]['source']}")
    else:
        print("\n✅ No new skills needed!")

    print(f"\n📄 Full recommendations in: {args.output}")
    print("\nUse the skill-generator meta-skill to create recommended skills.")


if __name__ == "__main__":
    main()
