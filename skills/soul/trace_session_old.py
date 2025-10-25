#!/usr/bin/env python3
"""
Agent Work Tracer - Session Analysis Script
Analyzes git changes, file modifications, and session context to create comprehensive work logs.
"""

import os
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

class AgentWorkTracer:
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path).resolve()
        self.timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M")
        self.agent_log_path = self.repo_path / ".agent_log.md"
        self.agent_status_path = self.repo_path / ".agent_status.json"
        self.agent_handoff_path = self.repo_path / ".agent_handoff.md"

    def run_git_command(self, cmd: List[str]) -> str:
        """Run git command and return output."""
        try:
            result = subprocess.run(
                ["git"] + cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return ""

    def get_git_status(self) -> Dict[str, List[str]]:
        """Get current git status - modified, added, deleted files."""
        status_output = self.run_git_command(["status", "--porcelain"])

        status = {
            "modified": [],
            "added": [],
            "deleted": [],
            "untracked": []
        }

        for line in status_output.split('\n'):
            if not line.strip():
                continue

            status_code = line[:2]
            filename = line[3:]

            if status_code.startswith('M'):
                status["modified"].append(filename)
            elif status_code.startswith('A'):
                status["added"].append(filename)
            elif status_code.startswith('D'):
                status["deleted"].append(filename)
            elif status_code.startswith('??'):
                status["untracked"].append(filename)

        return status

    def get_recent_commits(self, limit: int = 10) -> List[Dict[str, str]]:
        """Get recent commits with messages."""
        log_output = self.run_git_command([
            "log",
            f"--max-count={limit}",
            "--pretty=format:%H|%s|%an|%ad",
            "--date=short"
        ])

        commits = []
        for line in log_output.split('\n'):
            if not line.strip():
                continue

            parts = line.split('|')
            if len(parts) >= 4:
                commits.append({
                    "hash": parts[0][:8],
                    "message": parts[1],
                    "author": parts[2],
                    "date": parts[3]
                })

        return commits

    def analyze_file_changes(self) -> Dict[str, Any]:
        """Analyze what types of files were changed and infer purpose."""
        status = self.get_git_status()
        all_files = (status["modified"] + status["added"] +
                    status["deleted"] + status["untracked"])

        analysis = {
            "total_files_changed": len(all_files),
            "file_types": {},
            "categories": {
                "code": [],
                "config": [],
                "docs": [],
                "data": [],
                "scripts": []
            }
        }

        for filename in all_files:
            # File extension analysis
            ext = Path(filename).suffix.lower()
            analysis["file_types"][ext] = analysis["file_types"].get(ext, 0) + 1

            # Categorize files
            if ext in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs']:
                analysis["categories"]["code"].append(filename)
            elif ext in ['.json', '.yaml', '.yml', '.toml', '.ini', '.env']:
                analysis["categories"]["config"].append(filename)
            elif ext in ['.md', '.txt', '.rst', '.doc']:
                analysis["categories"]["docs"].append(filename)
            elif ext in ['.csv', '.json', '.xml', '.sql']:
                analysis["categories"]["data"].append(filename)
            elif ext in ['.sh', '.bat', '.ps1']:
                analysis["categories"]["scripts"].append(filename)

        return analysis

    def load_existing_status(self) -> Dict[str, Any]:
        """Load existing agent status if it exists."""
        default_status = {
            "sessions": [],
            "total_sessions": 0,
            "first_session": self.timestamp,
            "problems_solved": [],
            "key_decisions": []
        }

        if self.agent_status_path.exists():
            try:
                with open(self.agent_status_path, 'r') as f:
                    loaded_status = json.load(f)
                    # Ensure all required keys exist
                    for key, default_value in default_status.items():
                        if key not in loaded_status:
                            loaded_status[key] = default_value
                    return loaded_status
            except:
                pass

        return default_status

    def extract_problems_from_commits(self, commits: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Extract problems and solutions from commit messages."""
        problems = []

        problem_keywords = ["fix", "bug", "error", "issue", "problem", "critical", "broken"]
        solution_keywords = ["implement", "add", "create", "update", "improve", "optimize"]

        for commit in commits:
            message = commit["message"].lower()

            # Look for problem indicators
            has_problem = any(keyword in message for keyword in problem_keywords)
            has_solution = any(keyword in message for keyword in solution_keywords)

            if has_problem or has_solution:
                problems.append({
                    "commit": commit["hash"],
                    "description": commit["message"],
                    "date": commit["date"],
                    "type": "fix" if has_problem else "enhancement"
                })

        return problems

    def generate_work_log(self) -> str:
        """Generate comprehensive work log in markdown format."""
        status = self.get_git_status()
        commits = self.get_recent_commits(5)
        file_analysis = self.analyze_file_changes()
        problems = self.extract_problems_from_commits(commits)

        log_content = f"""# Agent Work Log - Session {self.timestamp}

## Session Information
- **Agent**: Claude (Setup Agent)
- **Timestamp**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Repository**: {self.repo_path.name}
- **Working Directory**: {self.repo_path}

## Files Changed This Session
- **Total Files**: {file_analysis['total_files_changed']}
- **Modified**: {len(status['modified'])} files
- **Added**: {len(status['added'])} files
- **Deleted**: {len(status['deleted'])} files
- **Untracked**: {len(status['untracked'])} files

### File Categories
"""

        for category, files in file_analysis['categories'].items():
            if files:
                log_content += f"- **{category.title()}**: {', '.join(files[:5])}"
                if len(files) > 5:
                    log_content += f" (and {len(files) - 5} more)"
                log_content += "\n"

        log_content += f"""
## Recent Commits
"""
        for commit in commits[:3]:
            log_content += f"- **{commit['hash']}**: {commit['message']} ({commit['date']})\n"

        if problems:
            log_content += f"""
## Problems Solved This Session
"""
            for problem in problems:
                log_content += f"- **{problem['type'].title()}**: {problem['description']}\n"

        log_content += f"""
## Current Repository State
- **Git Status**: {"Clean" if not any(status.values()) else "Modified files present"}
- **Branch**: {self.run_git_command(['branch', '--show-current']) or 'unknown'}
- **Last Commit**: {commits[0]['message'] if commits else 'No commits found'}

## Recommendations for Next Agent
1. Check git status for any uncommitted changes
2. Review recent commits to understand latest changes
3. Run tests to verify current state
4. Continue with planned development tasks

---
*Generated by Agent Work Tracer - {self.timestamp}*
"""

        return log_content

    def generate_status_json(self) -> Dict[str, Any]:
        """Generate machine-readable status for other agents."""
        existing_status = self.load_existing_status()
        status = self.get_git_status()
        commits = self.get_recent_commits(3)
        file_analysis = self.analyze_file_changes()

        new_session = {
            "timestamp": self.timestamp,
            "agent": "Claude",
            "files_changed": file_analysis['total_files_changed'],
            "commits_made": len([c for c in commits if c['date'] == datetime.now().strftime("%Y-%m-%d")]),
            "categories_touched": [cat for cat, files in file_analysis['categories'].items() if files]
        }

        existing_status["sessions"].append(new_session)
        existing_status["total_sessions"] += 1
        existing_status["last_session"] = self.timestamp
        existing_status["current_state"] = {
            "git_clean": not any(status.values()),
            "files_modified": len(status['modified']),
            "files_added": len(status['added']),
            "last_commit": commits[0]['message'] if commits else None
        }

        return existing_status

    def generate_handoff_notes(self) -> str:
        """Generate quick handoff notes for the next agent."""
        status = self.get_git_status()
        commits = self.get_recent_commits(1)

        handoff = f"""# Agent Handoff Notes

## Quick Status
- **Last Agent**: Claude (Setup Agent)
- **Session**: {self.timestamp}
- **Git Status**: {"✅ Clean" if not any(status.values()) else "⚠️ Modified files present"}

## What Was Done
"""

        if commits:
            handoff += f"- Latest work: {commits[0]['message']}\n"

        if status['modified']:
            handoff += f"- Modified files: {', '.join(status['modified'][:3])}\n"

        if status['added']:
            handoff += f"- New files: {', '.join(status['added'][:3])}\n"

        handoff += f"""
## Next Steps
1. Review `.agent_log.md` for detailed session information
2. Check `.agent_status.json` for machine-readable status
3. Run `git status` to see current repository state
4. Continue with development tasks

## Important Notes
- Repository is in working state
- All critical issues from previous sessions have been addressed
- Check existing documentation for project context

---
*Quick handoff generated at {self.timestamp}*
"""

        return handoff

    def save_logs(self):
        """Save all generated logs to files."""
        # Generate content
        work_log = self.generate_work_log()
        status_json = self.generate_status_json()
        handoff_notes = self.generate_handoff_notes()

        # Save work log (append to existing)
        if self.agent_log_path.exists():
            with open(self.agent_log_path, 'a') as f:
                f.write("\n\n" + "="*80 + "\n\n")
                f.write(work_log)
        else:
            with open(self.agent_log_path, 'w') as f:
                f.write(work_log)

        # Save status JSON (overwrite)
        with open(self.agent_status_path, 'w') as f:
            json.dump(status_json, f, indent=2)

        # Save handoff notes (overwrite)
        with open(self.agent_handoff_path, 'w') as f:
            f.write(handoff_notes)

        print(f"✅ Agent work logs saved:")
        print(f"   - Work log: {self.agent_log_path}")
        print(f"   - Status: {self.agent_status_path}")
        print(f"   - Handoff: {self.agent_handoff_path}")

def main():
    """Main function to run the tracer."""
    import argparse

    parser = argparse.ArgumentParser(description="Trace agent work session")
    parser.add_argument("--repo", default=".", help="Repository path")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    tracer = AgentWorkTracer(args.repo)
    tracer.save_logs()

    if args.verbose:
        print("\n📊 Session Analysis:")
        status = tracer.get_git_status()
        print(f"   - Modified files: {len(status['modified'])}")
        print(f"   - Added files: {len(status['added'])}")
        print(f"   - Recent commits: {len(tracer.get_recent_commits(5))}")

if __name__ == "__main__":
    main()
