# Claude Skills Creation Template

This document provides the complete template for creating Claude skills programmatically.
Use this as reference when NEXUS needs to create skills for systems without native skill creation.

## Skill Structure

Every Claude skill requires this directory structure:

```
.claude/skills/skill-name/
├── SKILL.md           # Required: Skill definition and metadata
├── FORMS.md           # Optional: User configuration options
├── examples.md        # Optional: Usage examples
├── reference.md       # Optional: Technical documentation
└── scripts/           # Optional: Helper scripts
    ├── main_script.py
    └── helper.py
```

## SKILL.md Template

The SKILL.md file is **required** and must include YAML frontmatter:

```markdown
---
name: skill-name
description: Brief description of what this skill does and when Claude should use it.
---

# SKILL-NAME - Descriptive Title

One or two sentences describing the skill's purpose.

## What SKILL-NAME does for Claude

**Capability 1**: Description
**Capability 2**: Description
**Capability 3**: Description
**Capability 4**: Description

## When Claude activates SKILL-NAME

Claude automatically uses this skill when:
- Trigger condition 1
- Trigger condition 2
- Trigger condition 3
- Trigger condition 4

## How it works

1. **Step 1**: Description of first step
2. **Step 2**: Description of second step
3. **Step 3**: Description of third step
4. **Step 4**: Description of fourth step

## Core Capabilities

Detailed description of what the skill can do:
- Capability description 1
- Capability description 2
- Capability description 3

## Example Usage

Brief example of the skill in action.

## Configuration (optional)

If the skill accepts configuration, describe it here.

---
*Additional notes or attribution*
```

### SKILL.md Requirements

1. **YAML Frontmatter (Required)**:
   - `name`: Lowercase, hyphen-separated skill identifier
   - `description`: Single-line description (used in skill list)

2. **Clear Activation Triggers**:
   - Describe when Claude should use this skill
   - Be specific about keywords, contexts, or patterns

3. **Capability Documentation**:
   - What the skill does
   - How it helps the user
   - What problems it solves

## FORMS.md Template

Optional configuration file for user preferences:

```markdown
# SKILL-NAME Configuration Guide

## Configuration Options

### Option Category 1

**Choice A**: [ ] Description of choice A
**Choice B**: [ ] Description of choice B (recommended)
**Choice C**: [ ] Description of choice C

### Option Category 2

**Setting 1**: [ ] Description
**Setting 2**: [ ] Description (recommended)
**Setting 3**: [ ] Description

## Configuration File

Create `.skill_name_config.json` in project root:

```json
{
  "option_category_1": "choice_b",
  "option_category_2": "setting_2",
  "advanced_settings": {
    "feature_1": true,
    "feature_2": false,
    "threshold": 5
  }
}
```

## Preset Configurations

### Preset 1: Conservative
```json
{
  "option_category_1": "choice_a",
  "automation_level": "low"
}
```

### Preset 2: Balanced (Recommended)
```json
{
  "option_category_1": "choice_b",
  "automation_level": "medium"
}
```

### Preset 3: Aggressive
```json
{
  "option_category_1": "choice_c",
  "automation_level": "high"
}
```

## No Configuration Needed

**SKILL-NAME works automatically without any configuration.**
These settings are optional optimizations for specific use cases.
```

## examples.md Template

Optional file showing real usage examples:

```markdown
# SKILL-NAME Usage Examples

## Example 1: [Scenario Name]

**User**: "User's request or context"

**Claude with SKILL-NAME**:
- Action 1
- Action 2
- "Claude's response using the skill..."

**Result**: What was accomplished

## Example 2: [Another Scenario]

**User**: Description of situation

**Claude with SKILL-NAME**:
- Automatic detection of context
- Skill activation
- Specific actions taken
- "Claude's response..."

**Result**: Outcome

## Example 3: [Complex Workflow]

**Session 1**: Initial setup
**Session 2**: Skill automatically continues work
**Session 3**: Completion

**Result**: Multi-session benefit demonstrated

---

*Include 3-6 examples showing different aspects of the skill*
```

## reference.md Template

Optional technical documentation:

```markdown
# SKILL-NAME Technical Reference

## Activation Logic

SKILL-NAME activates when Claude detects:
- Pattern 1: Description
- Pattern 2: Description
- Pattern 3: Description

### Keywords and Triggers
- Keyword set 1: [list]
- Keyword set 2: [list]
- File patterns: [list]

## Internal Architecture

### Components
1. **Component 1**: Purpose and function
2. **Component 2**: Purpose and function
3. **Component 3**: Purpose and function

### Data Flow
```
Input → Processing → Action → Output
```

## Configuration System

### Configuration File Location
`.skill_name_config.json` in project root

### Configuration Schema
```json
{
  "property1": "type",
  "property2": {
    "nested": "structure"
  }
}
```

## Helper Scripts

### scripts/main_script.py
Purpose: Primary functionality
Usage: Automatically called by Claude

### scripts/helper.py
Purpose: Utility functions
Usage: Support for main script

## API Reference (if applicable)

Document any exposed functions, classes, or interfaces.

## Performance Considerations

- Memory usage
- Processing speed
- Best practices for efficiency

## Troubleshooting

### Issue 1
**Symptom**: Description
**Cause**: Explanation
**Solution**: Fix

### Issue 2
**Symptom**: Description
**Cause**: Explanation
**Solution**: Fix

---
*Version history and technical notes*
```

## Script Template (Python)

For skills that need helper scripts:

```python
#!/usr/bin/env python3
"""
SKILL-NAME Helper Script
Purpose: Brief description of what this script does.

This script is called automatically by Claude when the skill is activated.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

class SkillNameHelper:
    """Main class for SKILL-NAME functionality"""

    def __init__(self, config_path: Optional[str] = None):
        self.config = self.load_config(config_path)
        self.setup()

    def load_config(self, config_path: Optional[str] = None) -> Dict:
        """Load configuration from file or use defaults"""
        default_config = {
            "option1": "default_value",
            "option2": True,
            "option3": 5
        }

        if not config_path:
            config_path = ".skill_name_config.json"

        if os.path.exists(config_path):
            try:
                with open(config_path) as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                print(f"Warning: Could not load config: {e}")

        return default_config

    def setup(self):
        """Initialize the skill"""
        # Initialization logic here
        pass

    def main_function(self, input_data):
        """Main functionality of the skill"""
        # Core logic here
        result = self.process(input_data)
        return result

    def process(self, data):
        """Process data according to skill purpose"""
        # Processing logic
        return f"Processed: {data}"

    def cleanup(self):
        """Clean up resources if needed"""
        pass


def main():
    """Command-line interface"""
    import argparse

    parser = argparse.ArgumentParser(description="SKILL-NAME Helper")
    parser.add_argument("--config", help="Config file path")
    parser.add_argument("--input", help="Input data")

    args = parser.parse_args()

    helper = SkillNameHelper(args.config)

    if args.input:
        result = helper.main_function(args.input)
        print(result)

    return 0


if __name__ == "__main__":
    exit(main())
```

## Best Practices for Skill Creation

### 1. Clear Activation Triggers
- Be specific about when the skill should activate
- Use concrete keywords and patterns
- Avoid overly broad triggers that activate too often

### 2. User-Friendly Descriptions
- Write for users, not just for Claude
- Explain benefits, not just features
- Include real examples

### 3. Optional Configuration
- Skills should work without configuration
- Provide sensible defaults
- Make advanced features opt-in

### 4. Documentation
- Document all capabilities
- Provide troubleshooting guides
- Include usage examples

### 5. Naming Conventions
- Use descriptive, memorable names
- Lowercase with hyphens: `api-master`, `test-guardian`
- Avoid generic names like `helper` or `tool`

### 6. Integration
- Work well with other skills
- Don't conflict with existing functionality
- Complement Claude's native capabilities

## Skill Categories

### Development Skills
Tools for coding, building, and maintaining software
- Examples: api-master, test-guardian, code-refiner

### Productivity Skills
Workflow optimization and task management
- Examples: task-master, doc-genius, deploy-sage

### Analysis Skills
Data processing, reporting, and insights
- Examples: data-wizard, perf-optimizer

### Domain-Specific Skills
Specialized tools for particular technologies
- Examples: docker-sage, react-wizard, sql-optimizer

## Validation Checklist

Before deploying a skill, verify:

- [ ] SKILL.md exists with valid YAML frontmatter
- [ ] `name` field matches directory name
- [ ] `description` is clear and concise
- [ ] Activation triggers are specific
- [ ] Examples demonstrate key features
- [ ] Configuration is optional (if present)
- [ ] Scripts are executable (if present)
- [ ] Documentation is complete and accurate
- [ ] Skill doesn't conflict with existing skills
- [ ] Tested with target use cases

## Integration with NEXUS

NEXUS uses these templates to create skills automatically:

1. **Analyze patterns** from SOUL or PRD data
2. **Select appropriate template** based on domain
3. **Populate template** with detected patterns
4. **Generate complete skill package**
5. **Deploy to `.claude/skills/`**

This ensures all NEXUS-generated skills follow Claude Skills best practices.

---

*This template is maintained as part of NEXUS for cross-platform skill creation*
