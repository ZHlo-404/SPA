"""Unified training entry point.

The method-specific training implementations live in ``methods/``.  Keeping
the original CSP implementation there makes it possible to add the other
methods without changing the currently working training code.
"""

import argparse
import runpy
import sys
from pathlib import Path


METHOD_SCRIPTS = {
    "csp": "csp_train.py",
    "hpl": "hpl_train.py",
    "dfsp": "dfsp_train.py",
    "troika": "troika_train.py",
}


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--method", default="csp", choices=sorted(METHOD_SCRIPTS))
    args, remaining = parser.parse_known_args()

    script = Path(__file__).resolve().parent / "methods" / METHOD_SCRIPTS[args.method]
    if not script.exists():
        raise FileNotFoundError(f"Method training script not found: {script}")

    # The method script keeps its original CLI (including --cfg and config
    # overrides), so only --method is consumed by this dispatcher.
    sys.argv = [str(script)] + remaining
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
