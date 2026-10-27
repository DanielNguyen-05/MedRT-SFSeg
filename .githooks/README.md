# Commit attribution

This repository disables automatic Claude Code commit and PR credit in `.claude/settings.json`.
The `commit-msg` and `pre-push` hooks also reject Claude author, committer, or `Co-authored-by` metadata.
They do not edit messages or change authors automatically.

Hooks are enabled for the current checkout. After cloning on another machine, enable them with:

```bash
git config --local core.hooksPath .githooks
```

These are local checks. They do not run for GitHub's web editor or in other checkouts unless enabled there.
