import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from inspect_ai._util.environ import environ_var


@contextmanager
def tournament_trace_file(log_root: Path, phase: str) -> Iterator[None]:
    """Route Inspect trace logs into a writable tournament-local location."""
    existing_trace_file = os.environ.get("INSPECT_TRACE_FILE")
    if existing_trace_file is not None and existing_trace_file.strip() != "":
        yield
        return

    trace_dir = log_root / ".traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_file = trace_dir / f"{phase}-trace-{os.getpid()}.log"
    with environ_var("INSPECT_TRACE_FILE", trace_file.as_posix()):
        yield
