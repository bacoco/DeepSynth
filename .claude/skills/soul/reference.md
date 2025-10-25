# SOUL API Reference

## Scripts

### trace_session.py

Main session analysis script.

**Usage**: `python scripts/trace_session.py [options]`

**Options**:
- `--verbose` - Detailed output during analysis
- `--quiet` - Minimal output
- `--append` - Append to existing logs instead of overwriting
- `--focus-problems` - Emphasize problem-solving analysis
- `--universal-format` - Generate cross-model compatible logs
- `--config FILE` - Use custom configuration file

**Output files**:
- `.agent_log.md` - Comprehensive work log
- `.agent_status.json` - Machine-readable status

### handoff_generator.py

Generates handoff notes for future agents.

**Usage**: `python scripts/handoff_generator.py [options]`

**Options**:
- `--quick` - Generate brief handoff notes
- `--detailed` - Generate comprehensive handoff
- `--both` - Generate both quick and detailed versions
- `--cross-model` - Format for different AI models

**Output files**:
- `.agent_handoff.md` - Handoff notes for next agent

### install.sh

Universal installation script for different environments.

**Usage**: `bash scripts/install.sh [environment]`

**Environments**:
- `claude` - Claude-specific setup
- `gpt` - GPT-specific setup  
- `universal` - Cross-platform setup

## Configuration

### .soul_config.json

```json
{
  "log_level": "basic|detailed|verbose",
  "include_git_diffs": boolean,
  "max_log_entries": number,
  "custom_templates": boolean,
  "output_format": "markdown|json|both",
  "git_analysis": {
    "include_diffs": boolean,
    "max_diff_lines": number,
    "ignore_patterns": ["*.log", "*.tmp"]
  },
  "handoff_settings": {
    "include_context": boolean,
    "max_context_lines": number,
    "priority_keywords": ["TODO", "FIXME", "BUG"]
  }
}
```

## Output Files

### .agent_log.md

Comprehensive session log with:
- Git change analysis
- Problems solved
- Decisions made
- Code modifications
- Next steps identified

### .agent_status.json

Machine-readable status:
```json
{
  "session_id": "uuid",
  "timestamp": "iso-date",
  "git_status": "clean|dirty",
  "files_modified": ["file1.py", "file2.js"],
  "problems_solved": ["issue1", "issue2"],
  "next_steps": ["step1", "step2"],
  "completion_status": "in-progress|completed|blocked"
}
```

### .agent_handoff.md

Structured handoff notes:
- Current session summary
- Immediate next steps
- Context for continuation
- Known issues or blockers