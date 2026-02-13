import pytest

from inspect_ai._util.local_server import (
    execute_shell_command,
    process_output_map,
    wait_for_server,
)


def _cleanup_process(process) -> None:
    if process.poll() is None:
        process.terminate()
        process.wait(timeout=5)
    process_output_map.pop(process, None)


def test_wait_for_server_includes_recent_output_on_exit() -> None:
    process = execute_shell_command(
        [
            "python3",
            "-c",
            (
                "import sys,time;"
                "sys.stderr.write('critical startup failure\\n');"
                "sys.stderr.flush();"
                "time.sleep(0.2);"
                "raise SystemExit(1)"
            ),
        ]
    )

    try:
        with pytest.raises(RuntimeError) as excinfo:
            wait_for_server("http://localhost:1", process, timeout=5)
    finally:
        _cleanup_process(process)

    message = str(excinfo.value)
    assert "Server process exited unexpectedly with code 1" in message
    assert "Recent server output" in message
    assert "critical startup failure" in message


def test_wait_for_server_includes_recent_output_on_timeout() -> None:
    process = execute_shell_command(
        [
            "python3",
            "-c",
            (
                "import sys,time;"
                "sys.stderr.write('warming up model\\n');"
                "sys.stderr.flush();"
                "time.sleep(3)"
            ),
        ]
    )

    try:
        with pytest.raises(TimeoutError) as excinfo:
            wait_for_server("http://localhost:1", process, timeout=1)
    finally:
        _cleanup_process(process)

    message = str(excinfo.value)
    assert "Server did not become ready within timeout period (1 seconds)" in message
    assert "Recent server output" in message
    assert "warming up model" in message
