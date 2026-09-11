"""Command line entry point (runtime commands are installed in later iterations)."""

import argparse

from . import __version__


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniHarness: a framework-free Python agent")
    parser.add_argument("--version", action="version", version=__version__)
    parser.parse_args()
    parser.print_help()
