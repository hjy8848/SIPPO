#!/usr/bin/env python3
"""Minimal curl-compatible GET helper for launcher health checks."""

from __future__ import annotations

import sys
import urllib.error
import urllib.request


def main() -> int:
    args = [arg for arg in sys.argv[1:] if arg not in {"-f", "-s", "-S", "-fsS"}]
    if len(args) != 1 or not args[0].startswith(("http://", "https://")):
        print("This helper only supports: curl -fsS URL", file=sys.stderr)
        return 2
    try:
        with urllib.request.urlopen(args[0], timeout=5) as response:
            sys.stdout.buffer.write(response.read())
    except (OSError, urllib.error.URLError) as error:
        print(error, file=sys.stderr)
        return 22
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
