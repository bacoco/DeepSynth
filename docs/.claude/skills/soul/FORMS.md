# SOUL Configuration Guide

## Claude Memory Preferences

Configure how Claude uses SOUL for your specific projects:

### Memory Depth

**Basic Memory**: [ ] Essential context only (faster)
**Detailed Memory**: [ ] Comprehensive context (recommended)
**Deep Memory**: [ ] Full historical analysis (complex projects)

### Project Type Optimization

**Web Development**: [ ] Focus on components, APIs, deployment
**Data Science/ML**: [ ] Emphasize experiments, model decisions, data insights
**General Coding**: [ ] Balanced approach to all development aspects
**Research/Analysis**: [ ] Prioritize findings, methodologies, conclusions

### Cross-Session Continuity

**Standard Continuity**: [ ] Normal handoff between sessions
**Enhanced Continuity**: [ ] Detailed context preservation
**Minimal Continuity**: [ ] Basic status only (lightweight)

### Code Analysis Preferences

**Include Code Diffs**: [ ] Yes [ ] No
**Focus on Problems Solved**: [ ] Yes [ ] No
**Track Decision Rationale**: [ ] Yes [ ] No
**Monitor Architecture Changes**: [ ] Yes [ ] No

## Configuration File

Create `.soul_config.json` in your project root:

```json
{
  "memory_depth": "detailed",
  "project_type": "web_development",
  "session_continuity": "enhanced",
  "code_analysis": {
    "include_diffs": true,
    "focus_problems": true,
    "track_decisions": true,
    "monitor_architecture": false
  },
  "output_preferences": {
    "detail_level": "comprehensive",
    "include_context": true,
    "max_history_entries": 100
  }
}
```

## Preset Configurations

### Preset 1: Web Development
```json
{
  "memory_depth": "detailed",
  "project_type": "web_development",
  "session_continuity": "enhanced"
}
```

### Preset 2: Data Science
```json
{
  "memory_depth": "deep",
  "project_type": "data_science",
  "session_continuity": "enhanced",
  "code_analysis": {
    "track_experiments": true,
    "monitor_model_performance": true
  }
}
```

### Preset 3: Lightweight
```json
{
  "memory_depth": "basic",
  "session_continuity": "standard",
  "code_analysis": {
    "include_diffs": false
  }
}
```

## No Configuration Needed

**SOUL works automatically without any configuration.** These settings are optional optimizations for specific use cases.
