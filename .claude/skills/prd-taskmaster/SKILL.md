---
name: prd-taskmaster
description: Analyzes PRD documents and task lists to identify which tasks need specialized skills, then generates NEXUS directives to create those skills. Claude activates this when working with project requirements and task planning.
---

# PRD-TASKMASTER - Requirements & Task Intelligence

Automatically analyzes PRDs and task lists to identify skill needs and generate creation directives.

## What PRD-TASKMASTER does for Claude

**PRD Analysis**: Parses project requirements documents and extracts task patterns
**Task Clustering**: Groups similar tasks that could benefit from specialized skills
**Skill Gap Detection**: Identifies which tasks lack appropriate skill support
**Directive Generation**: Creates NEXUS directives for needed skills automatically

## When Claude activates PRD-TASKMASTER

Claude automatically uses this skill when:
- User provides a PRD or project requirements document
- Working with task lists or project planning documents
- User mentions "todo", "tasks", or "requirements"
- Detecting markdown files with task patterns (todo.md, tasks.md, PRD.md)
- User asks "what skills do I need for this project?"

## How PRD-TASKMASTER works

1. **Document Detection**: Finds PRD and task list files automatically
2. **Task Extraction**: Parses tasks, requirements, and acceptance criteria
3. **Pattern Analysis**: Groups tasks by domain (API work, testing, deployment, etc.)
4. **Skill Mapping**: Determines which task groups need specialized skills
5. **Directive Creation**: Generates NEXUS directives for each skill need
6. **Implementation Plan**: Suggests order of skill creation based on task dependencies

## Task Pattern Recognition

PRD-TASKMASTER detects:
- API integration tasks → Suggests api-master skill
- Testing requirements → Suggests test-guardian skill
- Deployment tasks → Suggests deploy-sage skill
- Documentation needs → Suggests doc-genius skill
- Data processing → Suggests data-wizard skill
- Performance requirements → Suggests perf-optimizer skill

## Example workflow

```
1. User uploads PRD with 25 tasks
2. PRD-TASKMASTER analyzes tasks
3. Detects:
   - 8 tasks involve API work
   - 5 tasks involve testing
   - 4 tasks involve deployment
4. Generates 3 NEXUS directives:
   - api-master-directive.md
   - test-guardian-directive.md
   - deploy-sage-directive.md
5. Claude creates these skills
6. Project execution uses specialized skills
```

## Configuration

PRD-TASKMASTER looks for these files automatically:
- `*PRD*.md`, `*prd*.md` - Product requirements
- `*TODO*.md`, `*todo*.md` - Task lists
- `*TASK*.md`, `*task*.md` - Task planning
- `*REQUIREMENTS*.md` - Requirements docs
- `*ROADMAP*.md` - Project roadmaps

## Integration with NEXUS

PRD-TASKMASTER generates directives that NEXUS can use:
- Outputs to `.claude/skill-directives/`
- Includes task-specific examples in directives
- Maps tasks to recommended skills
- Provides implementation priority ordering

PRD-TASKMASTER ensures you have the right skills for your project before you start coding.
