# PRD-TASKMASTER Usage Examples

## Example 1: New Project with PRD

**User**: "I have this PRD for a new API project"

**Claude with PRD-TASKMASTER**:
- Automatically detects PRD file
- Extracts 25 tasks from document
- Analyzes task patterns
- "I found your PRD with 25 tasks. Let me analyze what skills you'll need..."
- Generates 3 skill directives: api-master, test-guardian, doc-genius
- "I've created directives for 3 specialized skills to help with this project"

## Example 2: Large TODO List

**User**: Provides todo.md with 50 tasks

**Claude with PRD-TASKMASTER**:
- Parses all 50 tasks
- Clusters into domains:
  - 12 API tasks
  - 8 deployment tasks
  - 6 testing tasks
  - 10 data processing tasks
- Recommends 4 skills in priority order
- Creates directives for each
- "Based on your task list, I recommend creating these skills first: api-master, data-wizard, deploy-sage, test-guardian"

## Example 3: Sprint Planning

**User**: "Help me plan this sprint" (provides sprint-tasks.md)

**Claude with PRD-TASKMASTER**:
- Analyzes sprint tasks
- Identifies skill needs per task
- Groups tasks by recommended skill
- "Here's how to organize your sprint:
  - Days 1-2: API integration tasks (use api-master)
  - Days 3-4: Data processing (use data-wizard)
  - Day 5: Testing (use test-guardian)"

## Example 4: Detecting Missing Skills

**User**: Working on complex project, mentions struggling with repetitive tasks

**Claude with PRD-TASKMASTER**:
- Reads existing PRD and completed tasks from SOUL
- Compares with available skills
- Identifies gaps
- "I notice you don't have a deployment skill, but 7 of your remaining tasks involve Docker and CI/CD. Let me create deploy-sage skill for you..."

## Example 5: Multi-Domain Project

**PRD contains**:
- Frontend requirements
- Backend API specs
- Database schema
- Deployment requirements
- Security requirements

**Claude with PRD-TASKMASTER**:
- Detects 5 distinct domains
- Generates 5 skill directives
- Prioritizes by task frequency and dependencies
- Creates implementation roadmap
- "This project needs 5 specialized skills. I recommend creating them in this order: 1) api-master (12 tasks), 2) db-wizard (8 tasks), 3) security-shield (7 tasks)..."

## Example 6: Integration with NEXUS

**Complete Workflow**:
```
1. User provides PRD
2. PRD-TASKMASTER analyzes → generates analysis report
3. NEXUS reads analysis report
4. NEXUS creates skill directives
5. Claude creates skills from directives
6. Skills available for project execution
```

**Result**: Fully automated skill provisioning from requirements to implementation
