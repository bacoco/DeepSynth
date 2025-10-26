---
name: nexus
description: Meta-skill that analyzes SOUL memory patterns and generates skill creation directives to guide Claude in building optimal skills. Claude activates this when detecting workflow optimization opportunities.
---

# NEXUS - Skill Strategy Generator

Analyzes SOUL data to detect patterns and generates directives that guide Claude's native skill creation.

## What NEXUS does for Claude

**Pattern Recognition**: Analyzes SOUL logs to detect recurring tasks and inefficiencies
**Directive Generation**: Creates focused markdown directives for skill creation
**Intelligence Amplification**: Provides Claude with actionable skill recommendations
**SOUL Integration**: Leverages persistent memory to identify optimization opportunities

## When Claude activates NEXUS

Claude automatically uses NEXUS when:
- Analyzing SOUL logs reveals repetitive patterns
- Multiple sessions show similar challenges or workflows
- User requests skill optimization or automation
- PRD or task analysis suggests specialized tooling needs
- Pattern frequency exceeds configured thresholds

## How NEXUS works

1. **SOUL Analysis**: Reads `.agent_log.md` and `.agent_status.json`
2. **Pattern Detection**: Uses `pattern_detector.py` to find recurring themes
3. **Directive Generation**: Creates skill creation guidance in `.claude/skill-directives/`
4. **Claude Execution**: Claude reads directive and creates skill using native capabilities

## NEXUS outputs DIRECTIVES, not full skills

**Key Difference**: NEXUS generates guidance files that help Claude create skills, rather than generating complete skill packages.

**Directive Contents:**
- Pattern analysis results (frequency, impact scores)
- Recommended skill name and purpose
- Key capabilities needed
- Example use cases from SOUL data
- Implementation guidance for Claude

## Example NEXUS workflow

```
1. SOUL logs show 15 API-related issues across 8 sessions
2. pattern_detector.py identifies "api_optimization" pattern
3. directive_generator.py creates:
   .claude/skill-directives/api-master-directive.md
4. Claude reads directive and creates api-master skill
5. New skill immediately available
```

## NEXUS scripts

**pattern_detector.py**: Sophisticated SOUL analysis engine
**directive_generator.py**: Creates skill creation guidance files
**install.sh**: Sets up NEXUS environment

NEXUS amplifies Claude's intelligence by providing data-driven skill recommendations.