# SOUL Technical Reference

## Claude's Automatic Behavior

### Memory Activation Triggers

Claude automatically activates SOUL when:
- Detecting existing `.agent_*` files in project
- Working on multi-session projects
- User mentions "continue", "previous work", or similar
- Complex problem-solving requiring context tracking
- Significant development milestones reached

### Memory Files Claude Creates

#### .agent_log.md
**Purpose**: Complete session history and context
**Content**:
- Work accomplished in each session
- Problems encountered and solutions
- Architectural decisions and rationale
- Code changes with context
- Learning and insights gained

#### .agent_status.json
**Purpose**: Machine-readable project state
**Structure**:
```json
{
  "session_id": "unique-identifier",
  "timestamp": "2024-10-25T13:30:00Z",
  "project_status": "active|paused|completed",
  "current_focus": "authentication system",
  "files_modified": ["auth.py", "login.js"],
  "problems_solved": ["OAuth integration", "session persistence"],
  "next_priorities": ["password reset", "2FA implementation"],
  "context_summary": "Building user authentication system..."
}
```

#### .agent_handoff.md
**Purpose**: Immediate context for next session
**Content**:
- Current session summary
- Immediate next steps
- Important context to remember
- Known issues or blockers
- Quick reference information

### Configuration Options

#### .soul_config.json (Optional)
```json
{
  "memory_depth": "basic|detailed|deep",
  "project_type": "web_development|data_science|general|research",
  "session_continuity": "minimal|standard|enhanced",
  "code_analysis": {
    "include_diffs": boolean,
    "focus_problems": boolean,
    "track_decisions": boolean,
    "monitor_architecture": boolean
  },
  "output_preferences": {
    "detail_level": "concise|comprehensive|verbose",
    "include_context": boolean,
    "max_history_entries": number
  }
}
```

## Cross-Model Compatibility

### Supported AI Models
- **Claude** (Sonnet, Haiku, Opus)
- **GPT** (3.5, 4, 4-turbo)
- **Codex** (GitHub Copilot integration)
- **Gemini** (Pro, Ultra)

### Universal Format
All memory files use markdown and JSON formats that work across different AI models, ensuring seamless handoffs between different AI systems.

## Internal Scripts (Claude manages automatically)

### trace_session.py
Analyzes git changes, extracts work patterns, creates comprehensive logs

### handoff_generator.py
Generates structured handoff notes optimized for AI consumption

### install.sh
Sets up SOUL in different AI environments

**Note**: Users never need to run these scripts manually - Claude handles everything automatically.
