---
name: soul
description: Creates persistent memory across AI agent sessions by automatically logging work, problems solved, and decisions made. Use when starting a new session, when development work is being completed, or when you need to document work for future agents.
---

# SOUL - AI Agent Memory System

Automatically creates work logs and handoff notes for seamless agent collaboration across sessions.

## Quick start

Document current session:
```bash
python trace_session.py --verbose
```

Generate handoff notes:
```bash
python handoff_generator.py --both
```

## Files created

- `.agent_log.md` - Complete work history
- `.agent_status.json` - Machine-readable status  
- `.agent_handoff.md` - Next steps for future agents

## Session workflow

Copy this checklist and track your progress:

```
Session Documentation:
- [ ] Step 1: Complete your development work
- [ ] Step 2: Run trace_session.py to analyze changes
- [ ] Step 3: Generate handoff notes
- [ ] Step 4: Verify files created successfully
```

**Step 1: Complete your development work**

Finish your coding, testing, or analysis tasks normally.

**Step 2: Run trace_session.py**

```bash
python trace_session.py --verbose
```

This analyzes git changes, extracts problems solved, and creates comprehensive logs.

**Step 3: Generate handoff notes**

```bash
python handoff_generator.py --both
```

Creates both quick and detailed handoff notes for the next agent.

**Step 4: Verify files created**

Check that these files exist:
- `.agent_log.md` (detailed history)
- `.agent_status.json` (structured data)
- `.agent_handoff.md` (immediate next steps)

## Advanced features

**Session analysis**: See [ANALYSIS.md](ANALYSIS.md) for git integration details
**Cross-model setup**: See [UNIVERSAL.md](UNIVERSAL.md) for GPT, Gemini, LLaMA integration  
**Templates**: See [TEMPLATES.md](TEMPLATES.md) for customizing log formats

## Configuration

Create `.soul_config.json` to customize:

```json
{
  "log_level": "detailed",
  "include_git_diffs": false,
  "max_log_entries": 50
}
```