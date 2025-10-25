#!/usr/bin/env python3
"""
NEXUS Skill Generator
Automatically creates new Claude skills based on detected patterns and needs.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
import argparse

class SkillGenerator:
    def __init__(self, config_path=None):
        self.config = self.load_config(config_path)
        self.skills_dir = Path(".claude/skills")
        self.templates = self.load_templates()
    
    def load_config(self, config_path):
        """Load NEXUS configuration or use defaults"""
        default_config = {
            "creation_sensitivity": "balanced",
            "skill_complexity": "comprehensive",
            "domain_focus": "all_domains",
            "naming_style": "creative",
            "pattern_threshold": "medium",
            "skill_integration": "connected",
            "learning_mode": "automatic"
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def load_templates(self):
        """Load skill templates for different domains"""
        return {
            "development": {
                "patterns": ["api", "framework", "testing", "deployment"],
                "capabilities": ["integration", "automation", "optimization"]
            },
            "productivity": {
                "patterns": ["task", "workflow", "organization", "time"],
                "capabilities": ["management", "tracking", "enhancement"]
            },
            "analysis": {
                "patterns": ["data", "report", "visualization", "insight"],
                "capabilities": ["processing", "generation", "recognition"]
            },
            "creative": {
                "patterns": ["content", "design", "documentation", "media"],
                "capabilities": ["generation", "assistance", "building"]
            }
        }
    
    def analyze_patterns(self, conversation_history):
        """Analyze conversation patterns to identify skill needs"""
        patterns = {
            "api_usage": self.count_pattern(conversation_history, ["api", "endpoint", "request", "response"]),
            "data_processing": self.count_pattern(conversation_history, ["data", "csv", "json", "process"]),
            "deployment": self.count_pattern(conversation_history, ["deploy", "docker", "ci/cd", "production"]),
            "documentation": self.count_pattern(conversation_history, ["readme", "docs", "documentation"]),
            "testing": self.count_pattern(conversation_history, ["test", "testing", "unit test", "integration"]),
            "workflow": self.count_pattern(conversation_history, ["workflow", "process", "automation"])
        }
        
        return {k: v for k, v in patterns.items() if v >= self.get_threshold()}
    
    def count_pattern(self, text, keywords):
        """Count occurrences of pattern keywords"""
        text_lower = text.lower()
        return sum(text_lower.count(keyword) for keyword in keywords)
    
    def get_threshold(self):
        """Get pattern detection threshold based on config"""
        thresholds = {"low": 2, "medium": 4, "high": 6}
        return thresholds.get(self.config["pattern_threshold"], 4)
    
    def generate_skill_name(self, pattern_type):
        """Generate creative skill name based on pattern"""
        naming_styles = {
            "creative": {
                "api_usage": "API-MASTER",
                "data_processing": "DATA-WIZARD", 
                "deployment": "DEPLOY-SAGE",
                "documentation": "DOC-GENIUS",
                "testing": "TEST-GUARDIAN",
                "workflow": "FLOW-OPTIMIZER"
            },
            "technical": {
                "api_usage": "API_HANDLER",
                "data_processing": "DATA_PROCESSOR",
                "deployment": "DEPLOY_MANAGER", 
                "documentation": "DOC_GENERATOR",
                "testing": "TEST_RUNNER",
                "workflow": "WORKFLOW_ENGINE"
            }
        }
        
        style = self.config.get("naming_style", "creative")
        return naming_styles.get(style, naming_styles["creative"]).get(pattern_type, "CUSTOM-SKILL")
    
    def create_skill(self, skill_name, pattern_type, detected_needs):
        """Create a complete skill package"""
        skill_dir = self.skills_dir / skill_name.lower()
        skill_dir.mkdir(parents=True, exist_ok=True)
        
        # Create scripts directory
        scripts_dir = skill_dir / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        
        # Generate skill files
        self.create_skill_md(skill_dir, skill_name, pattern_type, detected_needs)
        self.create_forms_md(skill_dir, skill_name, pattern_type)
        self.create_examples_md(skill_dir, skill_name, pattern_type)
        self.create_reference_md(skill_dir, skill_name, pattern_type)
        self.create_main_script(scripts_dir, skill_name, pattern_type)
        self.create_install_script(scripts_dir, skill_name)
        
        return skill_dir
    
    def create_skill_md(self, skill_dir, skill_name, pattern_type, detected_needs):
        """Generate main SKILL.md file"""
        content = f"""---
name: {skill_name.lower()}
description: Automatically handles {pattern_type} tasks based on detected usage patterns. Claude activates this when working with {pattern_type}-related challenges.
---

# {skill_name} - Intelligent {pattern_type.title()} Assistant

Automatically optimizes and handles {pattern_type} workflows based on your usage patterns.

## What {skill_name} does for Claude

**Pattern Recognition**: Detects {pattern_type} challenges and opportunities
**Automatic Optimization**: Streamlines {pattern_type} workflows
**Intelligent Assistance**: Provides context-aware {pattern_type} solutions
**Learning Enhancement**: Improves based on your specific {pattern_type} needs

## When Claude activates {skill_name}

Claude automatically uses this skill when:
- Working with {pattern_type}-related tasks
- Detecting {pattern_type} optimization opportunities
- Encountering {pattern_type} challenges or inefficiencies
- Building or maintaining {pattern_type} systems

## Core capabilities

**Smart Automation**: Handles repetitive {pattern_type} tasks
**Best Practices**: Applies industry standards and optimizations
**Error Prevention**: Anticipates and prevents common {pattern_type} issues
**Performance Optimization**: Ensures efficient {pattern_type} operations

## Automatic behavior

Claude uses {skill_name} transparently:
- Detects {pattern_type} needs automatically
- Applies optimizations without manual intervention
- Learns from your {pattern_type} preferences
- Adapts to your specific {pattern_type} workflows

Generated automatically by NEXUS on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        with open(skill_dir / "SKILL.md", 'w') as f:
            f.write(content)
    
    def create_forms_md(self, skill_dir, skill_name, pattern_type):
        """Generate FORMS.md configuration file"""
        content = f"""# {skill_name} Configuration Guide

## {pattern_type.title()} Optimization Preferences

Configure how Claude handles {pattern_type} tasks automatically:

### Automation Level

**Conservative**: [ ] Only automate clearly safe {pattern_type} tasks
**Balanced**: [ ] Automate most {pattern_type} tasks with confirmation (recommended)
**Aggressive**: [ ] Fully automate all {pattern_type} operations

### Performance Focus

**Speed**: [ ] Prioritize fast {pattern_type} operations
**Quality**: [ ] Emphasize {pattern_type} quality and reliability (recommended)
**Balance**: [ ] Optimize both speed and quality

### Learning Preferences

**Adaptive**: [ ] Learn and adapt to your {pattern_type} style (recommended)
**Standard**: [ ] Use industry best practices for {pattern_type}
**Custom**: [ ] Follow specific {pattern_type} guidelines you provide

## Configuration File

Create `.{skill_name.lower()}_config.json`:

```json
{{
  "automation_level": "balanced",
  "performance_focus": "quality", 
  "learning_mode": "adaptive",
  "{pattern_type}_preferences": {{
    "auto_optimization": true,
    "error_prevention": true,
    "best_practices": true
  }}
}}
```

## No Configuration Needed

**{skill_name} works automatically without configuration.** These settings are optional optimizations.

Generated automatically by NEXUS.
"""
        
        with open(skill_dir / "FORMS.md", 'w') as f:
            f.write(content)
    
    def create_examples_md(self, skill_dir, skill_name, pattern_type):
        """Generate examples.md file"""
        content = f"""# {skill_name} Automatic Behavior Examples

## Example 1: Automatic {pattern_type.title()} Detection

**User**: Working on {pattern_type}-related task

**Claude with {skill_name}**:
- Automatically detects {pattern_type} context
- Applies {skill_name} optimizations transparently
- Provides enhanced {pattern_type} assistance
- "I'm using {skill_name} to optimize this {pattern_type} workflow..."

## Example 2: Intelligent {pattern_type.title()} Assistance

**User**: Encounters {pattern_type} challenge

**Claude with {skill_name}**:
- Recognizes {pattern_type} problem pattern
- Automatically applies {skill_name} solutions
- Prevents common {pattern_type} issues
- "I've applied {skill_name} optimizations to handle this {pattern_type} challenge..."

## Example 3: Learning {pattern_type.Title()} Preferences

**Session 1**: Claude learns your {pattern_type} style
**Session 2**: {skill_name} adapts to your preferences
**Session 3**: Fully personalized {pattern_type} assistance
**Result**: "{skill_name} has learned your {pattern_type} preferences and will optimize accordingly"

## Example 4: Automatic {pattern_type.Title()} Optimization

**User**: Working on {pattern_type} project

**Claude with {skill_name}**:
- Continuously monitors {pattern_type} operations
- Applies real-time optimizations
- Suggests {pattern_type} improvements
- "I'm using {skill_name} to enhance your {pattern_type} workflow performance"

Generated automatically by NEXUS based on {pattern_type} patterns.
"""
        
        with open(skill_dir / "examples.md", 'w') as f:
            f.write(content)
    
    def create_reference_md(self, skill_dir, skill_name, pattern_type):
        """Generate reference.md technical documentation"""
        content = f"""# {skill_name} Technical Reference

## Claude's Automatic {pattern_type.Title()} Behavior

### Activation Triggers

Claude automatically activates {skill_name} when:
- Detecting {pattern_type}-related keywords and context
- Working with {pattern_type} files or systems
- User mentions {pattern_type} challenges or needs
- {pattern_type} optimization opportunities identified

### Core Capabilities

#### {pattern_type.Title()} Intelligence
- Pattern recognition for {pattern_type} tasks
- Automatic optimization application
- Best practice enforcement
- Error prevention and handling

#### Learning System
- Adapts to user {pattern_type} preferences
- Improves based on {pattern_type} usage patterns
- Personalizes {pattern_type} assistance
- Evolves {pattern_type} optimization strategies

### Configuration System

#### .{skill_name.lower()}_config.json Structure
```json
{{
  "automation_level": "conservative|balanced|aggressive",
  "performance_focus": "speed|quality|balance",
  "learning_mode": "adaptive|standard|custom",
  "{pattern_type}_preferences": {{
    "auto_optimization": boolean,
    "error_prevention": boolean,
    "best_practices": boolean,
    "custom_rules": []
  }}
}}
```

### Internal Scripts (Claude manages automatically)

#### {pattern_type}_optimizer.py
Handles {pattern_type} optimization and enhancement

#### {pattern_type}_analyzer.py  
Analyzes {pattern_type} patterns and usage

#### {pattern_type}_assistant.py
Provides intelligent {pattern_type} assistance

**Note**: Users never interact with these scripts directly - {skill_name} works automatically through Claude.

Generated automatically by NEXUS on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        with open(skill_dir / "reference.md", 'w') as f:
            f.write(content)
    
    def create_main_script(self, scripts_dir, skill_name, pattern_type):
        """Generate main functionality script"""
        content = f'''#!/usr/bin/env python3
"""
{skill_name} Main Script
Handles {pattern_type} optimization and automation.
Generated automatically by NEXUS.
"""

import json
import os
from datetime import datetime

class {skill_name.replace("-", "")}:
    def __init__(self):
        self.config = self.load_config()
        self.pattern_type = "{pattern_type}"
    
    def load_config(self):
        """Load skill configuration"""
        config_file = f".{skill_name.lower()}_config.json"
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)
        return self.get_default_config()
    
    def get_default_config(self):
        """Default configuration for {skill_name}"""
        return {{
            "automation_level": "balanced",
            "performance_focus": "quality",
            "learning_mode": "adaptive"
        }}
    
    def optimize_{pattern_type}(self, context):
        """Main {pattern_type} optimization function"""
        print(f"{{skill_name}} optimizing {{pattern_type}} context...")
        
        # Apply {pattern_type} optimizations based on config
        if self.config["automation_level"] == "aggressive":
            return self.full_optimization(context)
        elif self.config["automation_level"] == "conservative":
            return self.safe_optimization(context)
        else:
            return self.balanced_optimization(context)
    
    def full_optimization(self, context):
        """Aggressive {pattern_type} optimization"""
        return f"Full {pattern_type} optimization applied to: {{context}}"
    
    def safe_optimization(self, context):
        """Conservative {pattern_type} optimization"""
        return f"Safe {pattern_type} optimization applied to: {{context}}"
    
    def balanced_optimization(self, context):
        """Balanced {pattern_type} optimization"""
        return f"Balanced {pattern_type} optimization applied to: {{context}}"

if __name__ == "__main__":
    skill = {skill_name.replace("-", "")}()
    print(f"{{skill_name}} initialized for {{pattern_type}} optimization")
'''
        
        with open(scripts_dir / f"{pattern_type}_optimizer.py", 'w') as f:
            f.write(content)
        
        # Make script executable
        os.chmod(scripts_dir / f"{pattern_type}_optimizer.py", 0o755)
    
    def create_install_script(self, scripts_dir, skill_name):
        """Generate installation script"""
        content = f"""#!/bin/bash
# {skill_name} Installation Script
# Generated automatically by NEXUS

echo "Installing {skill_name}..."

# Check Python availability
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required for {skill_name}"
    exit 1
fi

# Install required packages (if any)
echo "Setting up {skill_name} environment..."

# Create config file if it doesn't exist
if [ ! -f ".{skill_name.lower()}_config.json" ]; then
    echo "Creating default configuration..."
    cat > .{skill_name.lower()}_config.json << EOF
{{
  "automation_level": "balanced",
  "performance_focus": "quality",
  "learning_mode": "adaptive"
}}
EOF
fi

echo "{skill_name} installation completed!"
echo "Claude will automatically use this skill when appropriate."
"""
        
        with open(scripts_dir / "install.sh", 'w') as f:
            f.write(content)
        
        # Make script executable
        os.chmod(scripts_dir / "install.sh", 0o755)
    
    def generate_skill_from_patterns(self, conversation_history):
        """Main function to generate skill from detected patterns"""
        patterns = self.analyze_patterns(conversation_history)
        
        if not patterns:
            print("No patterns detected that meet threshold for skill creation")
            return None
        
        # Select most prominent pattern
        primary_pattern = max(patterns.items(), key=lambda x: x[1])
        pattern_type, frequency = primary_pattern
        
        skill_name = self.generate_skill_name(pattern_type)
        
        print(f"Creating {skill_name} skill for {pattern_type} (detected {frequency} times)")
        
        skill_dir = self.create_skill(skill_name, pattern_type, patterns)
        
        print(f"Skill {skill_name} created successfully at {skill_dir}")
        return skill_dir

def main():
    parser = argparse.ArgumentParser(description="NEXUS Skill Generator")
    parser.add_argument("--config", help="Path to NEXUS configuration file")
    parser.add_argument("--pattern", help="Specific pattern to create skill for")
    parser.add_argument("--name", help="Custom skill name")
    parser.add_argument("--conversation", help="Path to conversation history file")
    
    args = parser.parse_args()
    
    generator = SkillGenerator(args.config)
    
    if args.pattern and args.name:
        # Create skill for specific pattern
        skill_dir = generator.create_skill(args.name, args.pattern, {args.pattern: 5})
        print(f"Skill {args.name} created at {skill_dir}")
    elif args.conversation:
        # Analyze conversation and create skill
        with open(args.conversation, 'r') as f:
            conversation = f.read()
        generator.generate_skill_from_patterns(conversation)
    else:
        print("Usage: python skill_generator.py --pattern <pattern> --name <name>")
        print("   or: python skill_generator.py --conversation <file>")

if __name__ == "__main__":
    main()
"""
        
        with open(scripts_dir / "skill_generator.py", 'w') as f:
            f.write(content)
        
        # Make script executable
        os.chmod(scripts_dir / "skill_generator.py", 0o755)