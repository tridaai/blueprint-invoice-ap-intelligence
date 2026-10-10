"""Module entry point.

    python -m ap_agent demo [--index N]   # traced single-invoice demo
    python -m ap_agent [--all|--index N]  # batch CLI over the samples
"""

from __future__ import annotations

import sys


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "demo":
        sys.argv = [sys.argv[0], *args[1:]]
        from .demo import main as demo_main

        demo_main()
        return
    from .cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
