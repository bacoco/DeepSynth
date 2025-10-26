#!/usr/bin/env python3
"""
NEXUS - SOUL Integration
Connects NEXUS to SOUL for pattern detection and skill generation.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Add SOUL to path
soul_scripts_path = Path(__file__).parent.parent.parent / "soul" / "scripts"
sys.path.insert(0, str(soul_scripts_path))

try:
    from soul_api import (
        get_pattern_analysis,
        get_soul_memory,
        add_soul_event,
        get_current_context
    )
    SOUL_AVAILABLE = True
except ImportError:
    SOUL_AVAILABLE = False
    print("⚠️  SOUL API not available. NEXUS will run in standalone mode.")


class SOULPatternDetector:
    """Détecte les patterns dans l'historique SOUL pour génération de skills"""

    def __init__(self, threshold: int = 5, window_days: int = 7):
        """
        Initialize pattern detector

        Args:
            threshold: Minimum occurrences to consider a pattern
            window_days: Number of days to analyze
        """
        self.threshold = threshold
        self.window_days = window_days

    def detect_patterns(self) -> Dict:
        """
        Detect patterns from SOUL memory

        Returns:
            Dict with detected patterns and recommendations
        """
        if not SOUL_AVAILABLE:
            return {"error": "SOUL not available", "patterns": {}}

        try:
            # Get pattern analysis from SOUL
            analysis = get_pattern_analysis(
                days=self.window_days,
                threshold=self.threshold
            )

            return analysis

        except Exception as e:
            print(f"❌ Error detecting patterns: {e}")
            return {"error": str(e), "patterns": {}}

    def get_skill_recommendations(self) -> List[Dict]:
        """
        Get skill recommendations based on SOUL patterns

        Returns:
            List of skill recommendations with priority
        """
        patterns = self.detect_patterns()

        if "error" in patterns:
            return []

        recommendations = []

        for pattern_type, pattern_info in patterns.get("patterns", {}).items():
            recommendation = {
                "pattern_type": pattern_type,
                "skill_name": pattern_info.get("suggested_skill", f"{pattern_type}-skill"),
                "priority": pattern_info.get("priority", "low"),
                "frequency": pattern_info.get("frequency", 0),
                "count": pattern_info.get("count", 0),
                "contexts": pattern_info.get("contexts", [])[:5],  # Top 5 exemples
                "reason": f"Detected {pattern_info.get('count', 0)} occurrences in {self.window_days} days"
            }

            recommendations.append(recommendation)

        # Sort by priority: critical > high > medium > low
        priority_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        recommendations.sort(
            key=lambda x: priority_order.get(x["priority"], 0),
            reverse=True
        )

        return recommendations

    def should_generate_skill(self, pattern_type: str, existing_skills: List[str]) -> bool:
        """
        Determine if a skill should be generated for this pattern

        Args:
            pattern_type: Type of pattern detected
            existing_skills: List of existing skill names

        Returns:
            True if skill should be generated
        """
        patterns = self.detect_patterns()

        if pattern_type not in patterns.get("patterns", {}):
            return False

        pattern_info = patterns["patterns"][pattern_type]
        suggested_skill = pattern_info.get("suggested_skill", "")

        # Don't generate if skill already exists
        if suggested_skill in existing_skills:
            return False

        # Only generate if priority is high or critical
        priority = pattern_info.get("priority", "low")
        if priority in ["high", "critical"]:
            return True

        # Or if frequency is very high (even if medium priority)
        frequency = pattern_info.get("frequency", 0)
        if frequency >= 2:  # 2+ times per day
            return True

        return False


class NEXUSSkillRegistry:
    """Registry of generated skills to avoid duplicates"""

    def __init__(self, skills_dir: Path = None):
        if skills_dir is None:
            skills_dir = Path(".claude/skills")
        self.skills_dir = Path(skills_dir)

    def get_existing_skills(self) -> List[str]:
        """Get list of existing skill names"""
        if not self.skills_dir.exists():
            return []

        skills = []
        for item in self.skills_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                skills.append(item.name)

        return skills

    def skill_exists(self, skill_name: str) -> bool:
        """Check if a skill already exists"""
        return skill_name in self.get_existing_skills()

    def register_skill_generation(self, skill_name: str, pattern_type: str, metadata: Dict = None):
        """
        Register that a skill was generated

        Records in SOUL memory for tracking
        """
        if SOUL_AVAILABLE:
            add_soul_event(
                "skill_generated",
                f"NEXUS generated skill: {skill_name}",
                {
                    "skill_name": skill_name,
                    "pattern_type": pattern_type,
                    "generated_by": "NEXUS",
                    **(metadata or {})
                }
            )


class NEXUSRecommendationEngine:
    """
    Engine that combines SOUL patterns with other signals
    to recommend skill generation
    """

    def __init__(self):
        self.pattern_detector = SOULPatternDetector()
        self.registry = NEXUSSkillRegistry()

    def get_all_recommendations(
        self,
        include_prd_recommendations: bool = True
    ) -> List[Dict]:
        """
        Get all skill recommendations from multiple sources

        Args:
            include_prd_recommendations: Whether to include PRD-TASKMASTER recs

        Returns:
            Combined list of skill recommendations
        """
        recommendations = []

        # Get SOUL-based recommendations
        soul_recs = self.pattern_detector.get_skill_recommendations()
        for rec in soul_recs:
            rec["source"] = "SOUL patterns"
            recommendations.append(rec)

        # TODO: Get PRD-TASKMASTER recommendations when integrated
        if include_prd_recommendations:
            # This will be implemented in Phase 6
            pass

        # Filter out existing skills
        existing_skills = self.registry.get_existing_skills()
        recommendations = [
            rec for rec in recommendations
            if rec["skill_name"] not in existing_skills
        ]

        return recommendations

    def get_top_recommendation(self) -> Optional[Dict]:
        """Get the highest priority recommendation"""
        recs = self.get_all_recommendations()
        if not recs:
            return None
        return recs[0]  # Already sorted by priority

    def record_skill_generated(self, skill_name: str, pattern_type: str):
        """Record that a skill was generated"""
        self.registry.register_skill_generation(skill_name, pattern_type)


def get_soul_patterns(threshold: int = 5, days: int = 7) -> List[Dict]:
    """
    Convenience function to get patterns from SOUL

    Args:
        threshold: Minimum pattern occurrences
        days: Days to analyze

    Returns:
        List of patterns with details
    """
    detector = SOULPatternDetector(threshold=threshold, window_days=days)
    return detector.get_skill_recommendations()


def should_generate_new_skills() -> bool:
    """
    Check if NEXUS should generate new skills based on SOUL analysis

    Returns:
        True if there are high-priority patterns requiring new skills
    """
    engine = NEXUSRecommendationEngine()
    recs = engine.get_all_recommendations()

    # Generate if there are any critical or high priority recommendations
    high_priority = [r for r in recs if r["priority"] in ["critical", "high"]]
    return len(high_priority) > 0


# Example usage
if __name__ == "__main__":
    print("NEXUS - SOUL Integration Test\n")

    if not SOUL_AVAILABLE:
        print("❌ SOUL not available. Install SOUL first.")
        sys.exit(1)

    # Test pattern detection
    detector = SOULPatternDetector(threshold=2, window_days=7)
    patterns = detector.detect_patterns()

    print(f"✓ Detected {patterns.get('patterns_detected', 0)} patterns")

    # Test recommendations
    recommendations = detector.get_skill_recommendations()
    print(f"✓ Generated {len(recommendations)} skill recommendations")

    for rec in recommendations:
        print(f"\n  {rec['skill_name']}:")
        print(f"    Priority: {rec['priority']}")
        print(f"    Frequency: {rec['frequency']:.1f}/day")
        print(f"    Count: {rec['count']} occurrences")

    # Test registry
    registry = NEXUSSkillRegistry()
    existing = registry.get_existing_skills()
    print(f"\n✓ Found {len(existing)} existing skills: {', '.join(existing)}")

    # Test recommendation engine
    engine = NEXUSRecommendationEngine()
    all_recs = engine.get_all_recommendations()
    print(f"\n✓ Total recommendations after filtering: {len(all_recs)}")

    if all_recs:
        top = engine.get_top_recommendation()
        print(f"\n🎯 Top recommendation: {top['skill_name']} ({top['priority']})")

    print("\n✓ NEXUS-SOUL integration test complete!")
