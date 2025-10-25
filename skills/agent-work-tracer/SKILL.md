# Agent Work Tracer

Automatically creates comprehensive work logs and handoff notes for AI agents working on development projects. This skill ensures continuity between agent sessions by documenting problems solved, decisions made, and current project state.

## When to use

This skill activates automatically when:
- Files have been modified in a git repository
- Technical problems have been solved during a session
- Development work is being completed
- An agent needs to document their work for future agents

## What it does

1. **Analyzes Session Changes**
   - Scans git status for modified/created files
   - Identifies new commits and their messages
   - Detects configuration changes and updates

2. **Documents Problem-Solution Pairs**
   - Extracts technical issues encountered
   - Records solutions implemented
   - Notes decision rationale and alternatives considered

3. **Creates Agent Communication Files**
   - `.agent_log.md` - Detailed human-readable work log
   - `.agent_status.json` - Machine-readable status for other agents
   - Updates existing logs with new session information

4. **Generates Handoff Notes**
   - Current project state summary
   - Recommended next actions
   - Important context for future agents
   - User preferences and requirements

## Files created

- `.agent_log.md` - Comprehensive work log with timestamps
- `.agent_status.json` - Structured data for agent-to-agent communication
- `.agent_handoff.md` - Quick reference for next agent

## Usage

The skill runs automatically when relevant. You can also invoke it manually:

```
Please use the agent work tracer to document this session.
```

## Benefits

- **Continuity**: Future agents understand previous work immediately
- **Efficiency**: No rediscovering of already-solved problems  
- **Collaboration**: Multiple agents can work on same project seamlessly
- **Accountability**: Clear record of what each agent accomplished
- **Learning**: Accumulated knowledge improves over time

## Example Output

The skill generates logs like:

```markdown
# Agent Work Log - Session 2024-10-25-14:30

## Agent: Claude (Setup Agent)
## Duration: ~2 hours
## Repository: deepseek-synthesia

### Problems Solved
1. **Disk Space Management**
   - Issue: Pipeline failing with "No space left on device"
   - Solution: Modified uploader to delete files after upload
   - Files: `efficient_incremental_uploader.py`

### Key Decisions
- Chose global pipeline over local-only approach
- Increased batch size to 10,000 for efficiency
- Used DejaVu Sans font for multilingual support

### Current State
- Global pipeline implemented and tested
- Cross-computer resume functionality working
- All critical bugs resolved

### Next Agent Should
- Run `./run_global_pipeline.sh` to continue dataset creation
- Monitor HuggingFace dataset progress
- Complete any remaining setup steps
```

## Configuration

The skill can be customized by creating a `.agent_tracer_config.json`:

```json
{
  "log_level": "detailed",
  "include_file_diffs": false,
  "max_log_entries": 50,
  "auto_commit_logs": true
}
```