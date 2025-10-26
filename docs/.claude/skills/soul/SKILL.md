---
name: soul
description: Gives Claude persistent memory across sessions by automatically creating work logs and handoff notes. Claude activates this skill when working on projects that need continuity between sessions.
---

# SOUL - Claude Memory Skill

Automatically provides Claude with persistent memory and session continuity.

## What SOUL does for Claude

**Automatic Memory**: Claude remembers what was accomplished in previous sessions
**Session Continuity**: Seamless transition between different Claude conversations  
**Work Documentation**: Automatic logging of problems solved and decisions made
**Context Preservation**: Maintains project context across time gaps

## When Claude activates SOUL

Claude automatically uses this skill when:
- Starting work on an existing project
- Completing significant development work
- Switching between different aspects of a project
- Needing to understand previous decisions or solutions

## How it works

1. **Session Start**: Claude checks for existing work logs and context
2. **During Work**: Claude tracks problems solved and decisions made  
3. **Session End**: Claude creates comprehensive handoff notes
4. **Next Session**: New Claude instance reads previous context and continues seamlessly

## Files Claude creates

- `.agent_log.md` - Complete work history and context
- `.agent_status.json` - Machine-readable project status
- `.agent_handoff.md` - Immediate next steps and context

## Claude's automatic behavior

**No manual commands needed** - Claude handles everything automatically:
- Detects when memory/continuity is needed
- Analyzes git changes and project state
- Creates appropriate documentation
- Reads previous session context when starting

## Configuration (optional)

Claude can use custom settings via `.soul_config.json`:

```json
{
  "memory_depth": "detailed",
  "include_code_analysis": true,
  "session_continuity": "enhanced"
}
```

SOUL works seamlessly with Claude, GPT, Codex, and Gemini.