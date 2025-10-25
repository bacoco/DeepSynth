# Session Analysis Details

## Git integration

SOUL analyzes git repositories to understand what changed during your session.

### Supported operations

- **Modified files**: Tracks which files were edited
- **New files**: Identifies files added to the repository  
- **Commits**: Extracts commit messages and timestamps
- **Branch status**: Records current branch and remote status

### Analysis output

```json
{
  "files_changed": {
    "modified": ["src/main.py", "config.json"],
    "added": ["tests/test_new_feature.py"],
    "deleted": []
  },
  "commits": [
    {
      "hash": "abc123",
      "message": "Add user authentication",
      "date": "2024-10-25"
    }
  ]
}
```

### Problem extraction

SOUL identifies problems and solutions from commit messages:

- **Keywords**: "fix", "bug", "error", "issue" → Problems
- **Keywords**: "add", "implement", "create" → Solutions
- **Context**: Links problems to specific files changed

### Non-git repositories

For projects without git, SOUL:
- Scans for recently modified files
- Uses file timestamps for chronology
- Creates basic change summaries