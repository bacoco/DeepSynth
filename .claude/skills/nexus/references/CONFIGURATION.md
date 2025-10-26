# NEXUS Configuration Reference

Complete configuration options for NEXUS.

## Configuration File

Create `.nexus_config.json` in your project root:

```json
{
  "analysis": {
    "threshold": 5,
    "window_days": 7,
    "auto_threshold": "high"
  },
  "sources": {
    "soul_memory": true,
    "prd_files": true,
    "task_lists": true,
    "code_analysis": false
  },
  "patterns": {
    "custom_patterns": {},
    "ignore_patterns": [],
    "priority_overrides": {}
  },
  "output": {
    "file": "NEXUS_RECOMMENDATIONS.md",
    "format": "markdown",
    "include_examples": true,
    "include_contexts": 5
  },
  "auto_generation": {
    "enabled": true,
    "max_skills_per_run": 5,
    "log_to_soul": true,
    "dry_run": false
  },
  "skill_generation": {
    "template": "default",
    "include_soul_api": true,
    "progressive_disclosure": true
  },
  "paths": {
    "skills_dir": ".claude/skills",
    "prd_patterns": ["*PRD*.md", "*REQUIREMENTS*.md", "*ROADMAP*.md"],
    "todo_patterns": ["TODO.md", "TASKS.md", "*TODO*.md"]
  },
  "logging": {
    "enabled": true,
    "file": ".nexus_auto.log",
    "level": "INFO"
  }
}
```

## Configuration Sections

### Analysis Settings

Controls how NEXUS analyzes patterns.

```json
"analysis": {
  "threshold": 5,
  "window_days": 7,
  "auto_threshold": "high"
}
```

**Options:**

- **`threshold`** (int, default: 5)
  - Minimum pattern occurrences to generate recommendation
  - Lower = more sensitive (more skills recommended)
  - Higher = less sensitive (only frequent patterns)
  - Recommended range: 2-10

- **`window_days`** (int, default: 7)
  - Number of days to analyze in SOUL memory
  - Shorter window = recent patterns only
  - Longer window = strategic patterns
  - Recommended range: 3-30

- **`auto_threshold`** (string, default: "high")
  - Minimum priority to auto-generate skills
  - Options: "low" | "medium" | "high" | "critical"
  - "high" = conservative (only high/critical)
  - "medium" = moderate (medium and above)
  - "critical" = very conservative (only critical)

### Source Settings

Controls which sources NEXUS analyzes.

```json
"sources": {
  "soul_memory": true,    // Analyze SOUL patterns
  "prd_files": true,      // Analyze PRD requirements
  "task_lists": true,     // Analyze TODO files
  "code_analysis": false  // Analyze codebase (future)
}
```

**Options:**

- **`soul_memory`** (bool, default: true)
  - Enable SOUL memory pattern detection
  - Recommended: true (core feature)

- **`prd_files`** (bool, default: true)
  - Enable PRD file analysis
  - Set false if no PRD files in project

- **`task_lists`** (bool, default: true)
  - Enable TODO/task file analysis
  - Set false if using external task tracking

- **`code_analysis`** (bool, default: false)
  - Enable codebase analysis (future feature)
  - Currently not implemented

### Pattern Settings

Advanced pattern customization.

```json
"patterns": {
  "custom_patterns": {
    "pdf_processing": {
      "keywords": ["pdf", "extract", "parse pdf"],
      "suggested_skill": "pdf-master",
      "priority_multiplier": 1.5
    }
  },
  "ignore_patterns": ["debug", "test_event"],
  "priority_overrides": {
    "security": "critical"
  }
}
```

**Options:**

- **`custom_patterns`** (object, default: {})
  - Define custom pattern detection rules
  - Keys: pattern type names
  - Values: pattern configuration objects

- **`ignore_patterns`** (array, default: [])
  - Event types to ignore during analysis
  - Useful for excluding debug/test events

- **`priority_overrides`** (object, default: {})
  - Force specific priorities for pattern types
  - Keys: pattern types
  - Values: "low" | "medium" | "high" | "critical"

### Output Settings

Controls recommendation output format.

```json
"output": {
  "file": "NEXUS_RECOMMENDATIONS.md",
  "format": "markdown",
  "include_examples": true,
  "include_contexts": 5
}
```

**Options:**

- **`file`** (string, default: "NEXUS_RECOMMENDATIONS.md")
  - Output file path
  - Can be relative or absolute

- **`format`** (string, default: "markdown")
  - Output format
  - Options: "markdown" | "json"

- **`include_examples`** (bool, default: true)
  - Include example contexts in recommendations
  - Set false for more concise output

- **`include_contexts`** (int, default: 5)
  - Number of example contexts to include
  - 0 = no examples
  - Recommended range: 3-10

### Auto-Generation Settings

Controls automatic skill generation behavior.

```json
"auto_generation": {
  "enabled": true,
  "max_skills_per_run": 5,
  "log_to_soul": true,
  "dry_run": false
}
```

**Options:**

- **`enabled`** (bool, default: true)
  - Enable automatic skill generation
  - Set false to only generate recommendations

- **`max_skills_per_run`** (int, default: 5)
  - Maximum skills to generate in one run
  - Prevents generating too many skills at once
  - 0 = unlimited

- **`log_to_soul`** (bool, default: true)
  - Record skill generation events in SOUL
  - Recommended: true (enables tracking)

- **`dry_run`** (bool, default: false)
  - Analyze only, don't create skills
  - Set true for testing configuration

### Skill Generation Settings

Controls how generated skills are structured.

```json
"skill_generation": {
  "template": "default",
  "include_soul_api": true,
  "progressive_disclosure": true
}
```

**Options:**

- **`template`** (string, default: "default")
  - Skill template to use
  - Options: "default" | "minimal" | "advanced"

- **`include_soul_api`** (bool, default: true)
  - Auto-integrate SOUL API in generated skills
  - Recommended: true (enables pattern tracking)

- **`progressive_disclosure`** (bool, default: true)
  - Apply progressive disclosure principles
  - Creates references/ directory for details

### Path Settings

Controls file and directory locations.

```json
"paths": {
  "skills_dir": ".claude/skills",
  "prd_patterns": ["*PRD*.md", "*REQUIREMENTS*.md"],
  "todo_patterns": ["TODO.md", "*TODO*.md"]
}
```

**Options:**

- **`skills_dir`** (string, default: ".claude/skills")
  - Directory where skills are stored

- **`prd_patterns`** (array, default: ["*PRD*.md", "*REQUIREMENTS*.md", "*ROADMAP*.md"])
  - Glob patterns for PRD files

- **`todo_patterns`** (array, default: ["TODO.md", "TASKS.md", "*TODO*.md"])
  - Glob patterns for TODO files

### Logging Settings

Controls log output.

```json
"logging": {
  "enabled": true,
  "file": ".nexus_auto.log",
  "level": "INFO"
}
```

**Options:**

- **`enabled`** (bool, default: true)
  - Enable logging

- **`file`** (string, default: ".nexus_auto.log")
  - Log file path

- **`level`** (string, default: "INFO")
  - Log level
  - Options: "DEBUG" | "INFO" | "WARNING" | "ERROR"

## Example Configurations

### Conservative (Fewer Skills)

```json
{
  "analysis": {
    "threshold": 10,
    "window_days": 14,
    "auto_threshold": "critical"
  },
  "auto_generation": {
    "max_skills_per_run": 2
  }
}
```

Only generates skills for very frequent patterns.

### Aggressive (More Skills)

```json
{
  "analysis": {
    "threshold": 2,
    "window_days": 3,
    "auto_threshold": "medium"
  },
  "auto_generation": {
    "max_skills_per_run": 10
  }
}
```

Generates skills for even infrequent patterns.

### SOUL-Only Analysis

```json
{
  "sources": {
    "soul_memory": true,
    "prd_files": false,
    "task_lists": false
  }
}
```

Only analyzes actual usage patterns from SOUL.

### Manual Mode (No Auto-Generation)

```json
{
  "auto_generation": {
    "enabled": false,
    "dry_run": true
  }
}
```

Only creates recommendations, never auto-generates.

### Custom Patterns

```json
{
  "patterns": {
    "custom_patterns": {
      "database_query": {
        "keywords": ["sql", "query", "database", "SELECT", "INSERT"],
        "suggested_skill": "db-optimizer",
        "priority_multiplier": 2.0
      },
      "image_processing": {
        "keywords": ["image", "resize", "convert", "png", "jpg"],
        "suggested_skill": "image-master",
        "priority_multiplier": 1.5
      }
    },
    "ignore_patterns": ["debug_log", "test_", "temp_"]
  }
}
```

### Security-Focused

```json
{
  "patterns": {
    "priority_overrides": {
      "security": "critical",
      "authentication": "critical",
      "authorization": "critical",
      "encryption": "high"
    }
  }
}
```

Forces security-related patterns to high/critical priority.

## Command-Line Override

Command-line arguments override config file:

```bash
# Config says threshold=5, this uses threshold=3
python auto_skill_generator.py --threshold 3

# Config says auto_threshold=high, this uses medium
python auto_skill_generator.py --auto-threshold medium
```

## Environment Variables

```bash
# Override config file location
export NEXUS_CONFIG="/path/to/custom_config.json"

# Override output file
export NEXUS_OUTPUT="/path/to/custom_output.md"

python auto_skill_generator.py
```

## Validation

Validate your configuration:

```bash
python .claude/skills/nexus/scripts/validate_config.py
```

Output:
```
✅ Configuration valid
   - threshold: 5 (valid range: 1-100)
   - window_days: 7 (valid range: 1-365)
   - auto_threshold: high (valid)
   - All paths exist
   - All patterns valid
```

## See Also

- [MANUAL_USAGE.md](MANUAL_USAGE.md) - Command-line options
- [ADVANCED.md](ADVANCED.md) - Advanced features
- [EXAMPLES.md](EXAMPLES.md) - Real-world examples
- Main SKILL.md - NEXUS overview
