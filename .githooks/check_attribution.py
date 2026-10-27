#!/usr/bin/env python3
"""Reject Claude commit attribution without changing commit messages or authors."""

from pathlib import Path
import re
import subprocess
import sys


BOT = re.compile(r"\bclaude\b|noreply@anthropic\.com", re.IGNORECASE)
COAUTHOR = re.compile(r"^[ \t]*Co-authored-by[ \t]*:(.*)$", re.IGNORECASE | re.MULTILINE)


def git(*args):
    return subprocess.check_output(["git", *args]).decode("utf-8", errors="replace")


def has_coauthor(message):
    return any(BOT.search(value) for value in COAUTHOR.findall(message))


def check_message(path):
    message = Path(path).read_text(encoding="utf-8", errors="replace")
    identities = [git("var", key) for key in ("GIT_AUTHOR_IDENT", "GIT_COMMITTER_IDENT")]
    return has_coauthor(message) or any(BOT.search(identity) for identity in identities)


def check_push(lines):
    checked = set()
    for line in lines:
        _local_ref, local_oid, _remote_ref, remote_oid = line.split()
        if not local_oid.strip("0"):
            continue  # Branch deletion introduces no commits.
        revision_args = [local_oid]
        if remote_oid.strip("0"):
            exists = subprocess.run(
                ["git", "cat-file", "-e", remote_oid + "^{commit}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode == 0
            if exists:
                revision_args += ["^" + remote_oid]
        for oid in git("rev-list", *revision_args).splitlines():
            if oid in checked:
                continue
            checked.add(oid)
            headers, message = git("cat-file", "commit", oid).split("\n\n", 1)
            identities = [value for value in headers.splitlines() if value.startswith(("author ", "committer "))]
            if has_coauthor(message) or any(BOT.search(value) for value in identities):
                print(f"Blocked Claude attribution in commit {oid[:12]}.", file=sys.stderr)
                return True
    return False


if __name__ == "__main__":
    blocked = check_message(sys.argv[2]) if sys.argv[1] == "message" else check_push(sys.stdin)
    if blocked:
        print("Commit/push rejected: remove Claude author or Co-authored-by metadata before retrying.", file=sys.stderr)
        sys.exit(1)
