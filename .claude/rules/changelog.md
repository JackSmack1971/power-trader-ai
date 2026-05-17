---
# Changelog Maintenance — Always Active

After every PR merges into `main`, update `CHANGELOG.md` before ending the session.

## Entry Format

Add under `## Detailed Change History`:

```markdown
#### [<short-sha>] <commit title>
**Date:** Month DD, YYYY
**Author:** <name>
**Pull Request:** #N (merged)
**Branch:** `branch-name` → `main`

- Bullet describing what changed and why.

**Files Added/Modified/Deleted:**
- `filename.py` (+N/-M lines) — one-line description
```

## Checklist

- [ ] New entry added under `## Detailed Change History`
- [ ] `**Generated:**` date updated at top of file
- [ ] `## Project Timeline` phase entry updated if a new phase was completed
- [ ] `## Component Evolution` updated if new scripts/services were added
- [ ] Changelog commit pushed to same branch (or a dedicated changelog PR if branch is gone)
