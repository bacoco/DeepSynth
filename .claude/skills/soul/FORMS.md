# SOUL Configuration Forms

## Session Configuration Form

Use this template to configure SOUL for your specific needs:

### Basic Configuration

**Project Type**: [ ] Web App [ ] ML/AI [ ] CLI Tool [ ] Library [ ] Other: ________

**Logging Level**: [ ] Basic [ ] Detailed [ ] Verbose

**Git Integration**: [ ] Include diffs [ ] Summary only [ ] Disabled

**Output Format**: [ ] Markdown only [ ] JSON only [ ] Both formats

### Advanced Settings

**Maximum Log Entries**: _______ (default: 50)

**Custom Templates**: [ ] Enabled [ ] Disabled

**Cross-Model Compatibility**: [ ] Enabled [ ] Disabled

**File Patterns to Ignore**:
- [ ] *.log
- [ ] *.tmp  
- [ ] node_modules/
- [ ] __pycache__/
- [ ] Custom: ________________

### Handoff Preferences

**Handoff Detail Level**: [ ] Quick [ ] Detailed [ ] Both

**Include Context**: [ ] Yes [ ] No

**Priority Keywords** (comma-separated): ________________________________

**Maximum Context Lines**: _______ (default: 100)

## Configuration Generation

Based on your selections above, create `.soul_config.json`:

```json
{
  "log_level": "YOUR_SELECTION",
  "include_git_diffs": true/false,
  "max_log_entries": YOUR_NUMBER,
  "custom_templates": true/false,
  "output_format": "YOUR_SELECTION",
  "git_analysis": {
    "include_diffs": true/false,
    "max_diff_lines": 500,
    "ignore_patterns": ["YOUR_PATTERNS"]
  },
  "handoff_settings": {
    "include_context": true/false,
    "max_context_lines": YOUR_NUMBER,
    "priority_keywords": ["YOUR_KEYWORDS"]
  }
}
```

## Quick Setup Forms

### Form 1: Basic Setup
For simple projects needing basic session tracking.

**Recommended settings**:
- Log level: Basic
- Git diffs: Summary only
- Output: Markdown only

### Form 2: Detailed Development
For complex projects with multiple agents.

**Recommended settings**:
- Log level: Detailed
- Git diffs: Include diffs
- Output: Both formats
- Cross-model: Enabled

### Form 3: Minimal Setup
For lightweight tracking.

**Recommended settings**:
- Log level: Basic
- Git diffs: Disabled
- Output: JSON only