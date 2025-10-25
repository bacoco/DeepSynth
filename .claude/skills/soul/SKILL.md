---
name: soul
description: Creates persistent memory across AI agent sessions by automatically logging work, problems solved, and decisions made. Use when starting a new session, when development work is being completed, or when you need to document work for future agents.
---

# SOUL - AI Agent Memory System

Automatically creates work logs and handoff notes for seamless agent collaboration across sessions.

## When to use

- Starting a new session (to understand previous work)
- Completing development work (to document progress)
- Switching between agents (to create handoffs)
- Need to track problems solved and decisions made

## Core functionality

**Session Documentation**: Analyzes git changes and creates comprehensive work logs
**Handoff Generation**: Creates structured notes for future agents
**Memory Persistence**: Maintains context across multiple AI sessions

## Quick workflow

1. Complete your development work
2. Run session analysis: `python scripts/trace_session.py --verbose`
3. Generate handoff: `python scripts/handoff_generator.py --both`
4. Verify files created: `.agent_log.md`, `.agent_status.json`, `.agent_handoff.md`

## Files created

- `.agent_log.md` - Complete work history with git analysis
- `.agent_status.json` - Machine-readable session status
- `.agent_handoff.md` - Next steps for future agents

## Configuration

Create `.soul_config.json` for customization:

```json
{
  "log_level": "detailed",
  "include_git_diffs": false,
  "max_log_entries": 50
}
```

See other documentation files for detailed guides and examples.