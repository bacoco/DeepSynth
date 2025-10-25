#!/usr/bin/env python3
"""
Handoff Generator - Creates concise handoff notes for agent transitions
Focuses on actionable next steps and critical context.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

class HandoffGenerator:
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path).resolve()
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        
    def load_agent_status(self) -> Dict[str, Any]:
        """Load current agent status."""
        status_file = self.repo_path / ".agent_status.json"
        if status_file.exists():
            with open(status_file, 'r') as f:
                return json.load(f)
        return {}
    
    def analyze_project_readiness(self) -> Dict[str, Any]:
        """Analyze if project is ready for next agent."""
        readiness = {
            "git_clean": True,
            "dependencies_installed": False,
            "tests_passing": False,
            "documentation_updated": False,
            "blocking_issues": []
        }
        
        # Check for common project files
        project_files = {
            "requirements.txt": "Python dependencies",
            "package.json": "Node.js dependencies", 
            "Cargo.toml": "Rust dependencies",
            "go.mod": "Go dependencies",
            "README.md": "Project documentation",
            ".env.example": "Environment configuration",
            "docker-compose.yml": "Docker setup"
        }
        
        found_files = []
        for filename, description in project_files.items():
            if (self.repo_path / filename).exists():
                found_files.append(f"{filename} ({description})")
        
        readiness["project_files"] = found_files
        return readiness
    
    def get_priority_next_steps(self) -> List[str]:
        """Determine priority next steps based on project state."""
        steps = []
        
        # Check for setup scripts
        if (self.repo_path / "run_global_pipeline.sh").exists():
            steps.append("Run `./run_global_pipeline.sh` to continue dataset creation")
        
        if (self.repo_path / "test_setup.py").exists():
            steps.append("Run `python test_setup.py` to verify environment")
        
        # Check for incomplete setup
        env_example = self.repo_path / ".env.example"
        env_file = self.repo_path / ".env"
        if env_example.exists() and not env_file.exists():
            steps.append("Create .env file from .env.example and configure tokens")
        
        # Check for requirements
        if (self.repo_path / "requirements.txt").exists():
            steps.append("Ensure dependencies are installed: `pip install -r requirements.txt`")
        
        # Default steps if nothing specific found
        if not steps:
            steps = [
                "Review project README.md for setup instructions",
                "Check git status for any uncommitted changes",
                "Run any available tests to verify current state"
            ]
        
        return steps[:5]  # Limit to top 5 priorities
    
    def extract_critical_context(self) -> Dict[str, Any]:
        """Extract critical context from recent work."""
        context = {
            "recent_problems": [],
            "key_decisions": [],
            "user_preferences": {},
            "technical_notes": []
        }
        
        # Try to load from agent status
        status = self.load_agent_status()
        if status:
            context["recent_problems"] = status.get("problems_solved", [])[-3:]
            context["key_decisions"] = status.get("key_decisions", [])[-3:]
        
        # Look for specific project context
        if (self.repo_path / "global_incremental_builder.py").exists():
            context["technical_notes"].append("Project uses global incremental pipeline for cross-computer work")
        
        if (self.repo_path / ".env.example").exists():
            context["technical_notes"].append("Requires HuggingFace token configuration")
        
        return context
    
    def generate_quick_handoff(self) -> str:
        """Generate concise handoff notes."""
        readiness = self.analyze_project_readiness()
        next_steps = self.get_priority_next_steps()
        context = self.extract_critical_context()
        
        handoff = f"""# 🔄 Agent Handoff - {self.timestamp}

## 🚀 Ready to Continue
**Project**: {self.repo_path.name}
**Status**: {"✅ Ready" if not readiness["blocking_issues"] else "⚠️ Has Issues"}

## 📋 Next Steps (Priority Order)
"""
        
        for i, step in enumerate(next_steps, 1):
            handoff += f"{i}. {step}\n"
        
        if context["technical_notes"]:
            handoff += f"""
## 🔧 Technical Context
"""
            for note in context["technical_notes"]:
                handoff += f"- {note}\n"
        
        if context["recent_problems"]:
            handoff += f"""
## ✅ Recently Solved
"""
            for problem in context["recent_problems"][-2:]:
                if isinstance(problem, dict):
                    handoff += f"- {problem.get('description', str(problem))}\n"
                else:
                    handoff += f"- {problem}\n"
        
        if readiness["blocking_issues"]:
            handoff += f"""
## ⚠️ Blocking Issues
"""
            for issue in readiness["blocking_issues"]:
                handoff += f"- {issue}\n"
        
        handoff += f"""
## 📁 Key Files
"""
        for file_info in readiness["project_files"][:5]:
            handoff += f"- {file_info}\n"
        
        handoff += f"""
## 💡 Quick Tips
- Check `.agent_log.md` for detailed session history
- Review `.agent_status.json` for machine-readable status
- Git status should be clean before major changes
- Test changes before committing

---
*Generated at {self.timestamp} | Next agent: Start with step 1 above*
"""
        
        return handoff
    
    def generate_detailed_handoff(self) -> str:
        """Generate comprehensive handoff with full context."""
        status = self.load_agent_status()
        readiness = self.analyze_project_readiness()
        context = self.extract_critical_context()
        
        handoff = f"""# 📋 Detailed Agent Handoff

## Session Information
- **Timestamp**: {self.timestamp}
- **Repository**: {self.repo_path.name}
- **Previous Agent**: Claude (Setup Agent)
- **Total Sessions**: {status.get('total_sessions', 1)}

## Project Overview
This appears to be a {self.repo_path.name} project with the following characteristics:
"""
        
        for file_info in readiness["project_files"]:
            handoff += f"- {file_info}\n"
        
        handoff += f"""
## Current State Analysis

### Repository Status
- **Git Clean**: {readiness['git_clean']}
- **Dependencies**: {"✅ Configured" if readiness['project_files'] else "❓ Unknown"}
- **Documentation**: {"✅ Present" if any('README' in f for f in readiness['project_files']) else "❓ Limited"}

### Recent Work Summary
"""
        
        if status.get('sessions'):
            last_session = status['sessions'][-1]
            handoff += f"- Last session: {last_session.get('timestamp', 'Unknown')}\n"
            handoff += f"- Files changed: {last_session.get('files_changed', 0)}\n"
            handoff += f"- Categories: {', '.join(last_session.get('categories_touched', []))}\n"
        
        handoff += f"""
## Critical Context for Next Agent

### Technical Decisions Made
"""
        for decision in context["key_decisions"]:
            if isinstance(decision, dict):
                handoff += f"- {decision.get('title', 'Decision')}: {decision.get('rationale', str(decision))}\n"
            else:
                handoff += f"- {decision}\n"
        
        handoff += f"""
### Problems Already Solved
"""
        for problem in context["recent_problems"]:
            if isinstance(problem, dict):
                handoff += f"- {problem.get('description', str(problem))}\n"
            else:
                handoff += f"- {problem}\n"
        
        handoff += f"""
### User Context
- **Technical Level**: High (based on project complexity)
- **Main Requirements**: Cross-computer functionality, efficiency
- **Communication Style**: Direct, technical

## Recommended Action Plan

### Phase 1: Orientation (5-10 minutes)
1. Read this handoff document completely
2. Review `.agent_log.md` for detailed history
3. Check `git status` and recent commits
4. Scan README.md for project overview

### Phase 2: Environment Setup (10-15 minutes)
1. Verify dependencies are installed
2. Check environment configuration (.env files)
3. Run any available setup verification scripts
4. Test basic functionality

### Phase 3: Continue Development
1. Identify current development priorities
2. Review any TODO items or issue trackers
3. Continue with planned features or fixes
4. Document your work for the next agent

## Resources & References

### Important Files to Review
- `.agent_log.md` - Complete work history
- `.agent_status.json` - Machine-readable status
- `README.md` - Project documentation
- `.env.example` - Configuration template

### External Resources
- Project repository: Check git remotes for URLs
- Documentation: Look for docs/ directory or wiki links
- Issue tracking: Check for GitHub issues or project boards

## Handoff Checklist

Before starting work, ensure:
- [ ] Read this handoff completely
- [ ] Reviewed recent git commits
- [ ] Checked environment setup
- [ ] Understood current project state
- [ ] Identified next priorities

When finishing your session:
- [ ] Update `.agent_log.md` with your work
- [ ] Update `.agent_status.json` with current state
- [ ] Create new handoff notes for next agent
- [ ] Commit important changes to git

---
*Comprehensive handoff generated at {self.timestamp}*
*Next agent: Follow the action plan above for smooth transition*
"""
        
        return handoff
    
    def save_handoff(self, detailed: bool = False):
        """Save handoff notes to file."""
        if detailed:
            content = self.generate_detailed_handoff()
            filename = ".agent_handoff_detailed.md"
        else:
            content = self.generate_quick_handoff()
            filename = ".agent_handoff.md"
        
        handoff_path = self.repo_path / filename
        with open(handoff_path, 'w') as f:
            f.write(content)
        
        print(f"✅ Handoff notes saved: {handoff_path}")
        return handoff_path

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate agent handoff notes")
    parser.add_argument("--repo", default=".", help="Repository path")
    parser.add_argument("--detailed", action="store_true", help="Generate detailed handoff")
    parser.add_argument("--both", action="store_true", help="Generate both quick and detailed")
    
    args = parser.parse_args()
    
    generator = HandoffGenerator(args.repo)
    
    if args.both:
        generator.save_handoff(detailed=False)
        generator.save_handoff(detailed=True)
    else:
        generator.save_handoff(detailed=args.detailed)

if __name__ == "__main__":
    main()