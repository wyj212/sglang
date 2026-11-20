"""Changed-line coverage for a patch, without the coverage package.

Answers one question: of the lines this adaptation *added*, which ones did a
test run actually execute?

* changed lines come straight from ``git diff -U0``;
* executed lines come from the per-pid dumps written by ``sitecustomize.py``;
* the denominator keeps only **logical line starts** - a tracer reports a
  multi-line statement on its first line only, and never reports a docstring
  body, so counting continuation lines as separate executable lines would
  systematically understate the result.

The rev pair is pinned rather than expressed as ``HEAD~12`` so the numbers stay
stable once test-only commits land on top of the adaptation.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import subprocess
import sys
import tokenize

HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")

DEFAULT_REV = "1392ba85e2~12"
DEFAULT_TO = "1392ba85e2"
DEFAULT_DUMPS = "/tmp/qguard_cov"


def git(*args: str, repo: str) -> str:
    return subprocess.run(
        ["git", "-C", repo, *args], capture_output=True, text=True, check=True
    ).stdout


def changed_line_numbers(repo: str, rev: str, to: str) -> dict[str, list[int]]:
    """Added line numbers per file, straight from the diff hunks."""
    out: dict[str, list[int]] = collections.defaultdict(list)
    current = None
    for line in git("diff", "-U0", rev, to, repo=repo).splitlines():
        if line.startswith("+++ b/"):
            current = line[6:]
            continue
        if line.startswith("+++ /dev/null"):
            current = None
            continue
        if line.startswith("@@"):
            match = HUNK.match(line)
            if match and current:
                start = int(match.group(1))
                count = int(match.group(2)) if match.group(2) is not None else 1
                out[current].extend(range(start, start + count))
    return dict(out)


def read_lines(repo: str, relpath: str) -> list[str]:
    with open(os.path.join(repo, relpath), encoding="utf-8") as fh:
        return fh.read().splitlines()


def statement_start_lines(repo: str, relpath: str) -> set[int]:
    """Line numbers where a *logical* line begins.

    The tracer reports a multi-line statement on its first line only, and a
    docstring as a single constant, so counting continuation lines as
    "changed executable lines" would understate the coverage.  tokenize gives
    exactly the set of lines that can carry a line event.
    """
    starts: set[int] = set()
    pending = None
    path = os.path.join(repo, relpath)
    try:
        with open(path, "rb") as fh:
            tokens = list(tokenize.tokenize(fh.readline))
    except (tokenize.TokenError, IndentationError, SyntaxError, OSError):
        # Never let an unparsable file silently vanish from the denominator.
        return set(range(1, len(read_lines(repo, relpath)) + 1))

    for tok in tokens:
        if tok.type in (tokenize.NL, tokenize.COMMENT):
            continue
        if tok.type == tokenize.NEWLINE:
            pending = None
        elif tok.type in (
            tokenize.INDENT,
            tokenize.DEDENT,
            tokenize.ENCODING,
            tokenize.ENDMARKER,
        ):
            continue
        elif pending is None:
            starts.add(tok.start[0])
            pending = tok.start[0]
    return starts


def load_executed(dumps: str) -> tuple[set[tuple[str, int]], int]:
    """Union every per-process dump; returns (rows, number of dump files)."""
    executed: set[tuple[str, int]] = set()
    files = sorted(glob.glob(os.path.join(dumps, "cov-*.data")))
    for path in files:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                filename, _, lineno = line.rstrip("\n").partition("\t")
                if lineno:
                    executed.add((filename, int(lineno)))
    return executed, len(files)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.getcwd())
    # The adaptation is a fixed range of twelve commits.  Pinning both ends
    # keeps the numbers stable once test commits land on top of it.
    parser.add_argument("--rev", default=DEFAULT_REV)
    parser.add_argument("--to", default=DEFAULT_TO)
    parser.add_argument("--dumps", default=DEFAULT_DUMPS)
    parser.add_argument("--exclude-prefix", default="test/")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    repo = os.path.abspath(args.repo)
    changed = changed_line_numbers(repo, args.rev, args.to)
    changed = {
        relpath: linenos
        for relpath, linenos in changed.items()
        if not relpath.startswith(args.exclude_prefix)
    }
    if not changed:
        print("no changed files in %s..%s" % (args.rev, args.to), file=sys.stderr)
        return 1

    executed, n_dumps = load_executed(args.dumps)
    if not executed:
        print("no execution dumps under %s" % args.dumps, file=sys.stderr)
        return 1

    effective: dict[str, set[int]] = {}
    for relpath, linenos in changed.items():
        starts = statement_start_lines(repo, relpath)
        keep = {n for n in linenos if n in starts}
        if keep:
            effective[relpath] = keep

    per_file = {}
    missed = {}
    total = hit_total = 0
    for relpath, linenos in sorted(effective.items()):
        absolute = os.path.join(repo, relpath)
        source = read_lines(repo, relpath)
        hit = {n for n in linenos if (absolute, n) in executed}
        total += len(linenos)
        hit_total += len(hit)
        per_file[relpath] = {
            "changed_effective": len(linenos),
            "executed": len(hit),
            "percent": round(100.0 * len(hit) / len(linenos), 1),
        }
        missing = sorted(linenos - hit)
        if missing:
            missed[relpath] = [
                {
                    "line": n,
                    "source": source[n - 1].rstrip() if n <= len(source) else "",
                }
                for n in missing
            ]

    report = {
        "rev": "%s..%s" % (args.rev, args.to),
        "coverage_dumps": n_dumps,
        "overall": {
            "changed_effective_lines": total,
            "executed": hit_total,
            "percent": round(100.0 * hit_total / total, 1) if total else 0.0,
        },
        "per_file": per_file,
        "missed": missed,
        "files_without_executable_changes": sorted(
            set(changed) - set(effective)
        ),
    }

    print("===== changed-line coverage =====")
    print(
        "total: %d/%d = %.1f%%   (%d process dump(s), %d executed line rows)"
        % (hit_total, total, report["overall"]["percent"], n_dumps, len(executed))
    )
    print()
    print("%-62s %6s %6s %8s" % ("file", "added", "hit", "coverage"))
    for relpath, stats in per_file.items():
        print(
            "%-62s %6d %6d %7.1f%%"
            % (
                relpath,
                stats["changed_effective"],
                stats["executed"],
                stats["percent"],
            )
        )
    if report["files_without_executable_changes"]:
        print()
        print("changed lines but no logical line start (continuations/docstrings):")
        for relpath in report["files_without_executable_changes"]:
            print("  %s" % relpath)

    print()
    print("===== missed changed lines =====")
    for relpath, rows in missed.items():
        print()
        print("-- %s  (%d)" % (relpath, len(rows)))
        for row in rows:
            print("   %5d  %s" % (row["line"], row["source"][:100]))

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        print()
        print("json -> %s" % args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
