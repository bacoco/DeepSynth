# Log Templates and Customization

## Default log structure

```markdown
# Agent Work Log - Session [TIMESTAMP]

## Session Information
- Agent: [AGENT_NAME]
- Repository: [REPO_NAME]
- Duration: [ESTIMATED_TIME]

## Problems Solved
1. **[PROBLEM_TITLE]**
   - Issue: [DESCRIPTION]
   - Solution: [HOW_FIXED]
   - Files: [FILES_CHANGED]

## Key Decisions
- [DECISION_1]: [RATIONALE]
- [DECISION_2]: [RATIONALE]

## Current State
- [STATUS_SUMMARY]
- [IMPORTANT_NOTES]

## Next Agent Should
- [ACTION_1]
- [ACTION_2]
```

## Customizing templates

Edit `log_template.md` to change the format:

```markdown
# Custom Template

## My Custom Sections
- Custom field 1
- Custom field 2

## Standard Sections
[Keep the sections you want]
```

## Configuration options

In `.soul_config.json`:

```json
{
  "template_file": "custom_template.md",
  "include_sections": [
    "problems_solved",
    "key_decisions", 
    "next_steps"
  ],
  "exclude_sections": [
    "file_analysis",
    "commit_details"
  ]
}
```

## Handoff note templates

Quick handoff format:
```markdown
# Agent Handoff

## Priority Actions
1. [MOST_IMPORTANT]
2. [SECOND_PRIORITY]

## Context
- [KEY_INFO]

## Status
- [CURRENT_STATE]
```

Detailed handoff includes:
- Complete project analysis
- Technical decisions made
- User preferences discovered
- Comprehensive action plan