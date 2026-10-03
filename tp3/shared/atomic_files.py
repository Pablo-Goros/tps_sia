"""Atomic file replacement that tolerates transient locks on Windows.

On Windows, os.replace fails with PermissionError while another process
(antivirus, search indexer, editor file watcher) holds the target open. Such
locks last milliseconds to seconds, so the replacement is retried with
increasing waits before the error is raised. Other errors are not retried.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

RETRY_DELAYS = (0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 1.0, 1.0, 1.0)  # ~5.5 s in total


def replace(source: str | Path, target: str | Path, delays=RETRY_DELAYS, sleep=time.sleep) -> None:
    """os.replace(source, target), retrying only on PermissionError."""
    for delay in delays:
        try:
            os.replace(source, target)
            return
        except PermissionError:
            sleep(delay)
    os.replace(source, target)
