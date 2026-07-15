"""Unified evaluation entry point."""

import argparse
import runpy
import sys
from pathlib import Path


METHOD_SCRIPTS = {
    "csp": "csp_eval.py",
    "hpl": "hpl_eval.py",
    "dfsp": "dfsp_eval.py",
    "troika": "troika_eval.py",
}


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--method", default="csp", choices=sorted(METHOD_SCRIPTS))
    args, remaining = parser.parse_known_args()

    script = Path(__file__).resolve().parent / "methods" / METHOD_SCRIPTS[args.method]
    if not script.exists():
        raise FileNotFoundError(f"Method evaluation script not found: {script}")

    sys.argv = [str(script)] + remaining
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
