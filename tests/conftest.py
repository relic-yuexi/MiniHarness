"""Keep test scratch data local, isolated, and out of the user's system Temp."""

import os
from pathlib import Path


def pytest_configure(config):
    if config.option.basetemp is None:
        root = config.rootpath / ".pytest_tmp"
        root.mkdir(exist_ok=True)
        config.option.basetemp = str(root / f"run-{os.getpid()}")
    else:
        Path(config.option.basetemp).parent.mkdir(parents=True, exist_ok=True)
