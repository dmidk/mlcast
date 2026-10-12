"""Tests for the console-log capture and its MLflow upload callback."""

from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pytorch_lightning.loggers import MLFlowLogger

from mlcast.callbacks import LogConsoleToMLflowCallback, _collapse_redraws, _ConsoleCapture


def _use_fd_backed_std_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point sys.stdout/sys.stderr at fds 1/2, as in a normal process.

    pytest's capture swaps them for objects that don't write to fds 1/2, so
    without this, Python-level writes would bypass the fd-level capture. Must
    be called from the test body: pytest re-installs its own streams when
    the call phase starts, which would undo this if done in a fixture.
    """
    for name, fd in (("stdout", 1), ("stderr", 2)):
        stream = io.TextIOWrapper(open(fd, "wb", closefd=False), line_buffering=True)
        monkeypatch.setattr(sys, name, stream)


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("plain", "plain"),
        ("\r 10%\r 20%\r 30%", " 30%"),
        ("abc\r", "abc\r"),
        ("abc\rde", "de"),
        ("x\r\r", "x\r"),
    ],
)
def test_collapse_redraws(line: str, expected: str) -> None:
    assert _collapse_redraws(line) == expected


def test_console_capture_records_all_streams(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Python-level writes, raw fd writes and progress-bar redraws all land in
    the file, with each redrawn line reduced to its final state."""
    _use_fd_backed_std_streams(monkeypatch)
    capture = _ConsoleCapture(tmp_path / "console.log")
    capture.start()
    try:
        print("from print")
        sys.stderr.write("from stderr\n")
        os.write(1, b"from raw fd 1\n")
        sys.stderr.write("\r 10%\r 50%\r100%\n")
        sys.stdout.write("unterminated")
    finally:
        capture.stop()

    lines = (tmp_path / "console.log").read_text().splitlines()
    assert sorted(lines) == sorted(["from print", "from stderr", "from raw fd 1", "100%", "unterminated"])


def _mlflow_trainer(is_global_zero: bool = True) -> MagicMock:
    trainer = MagicMock()
    trainer.is_global_zero = is_global_zero
    trainer.logger = MagicMock(spec=MLFlowLogger)
    trainer.logger.run_id = "run-123"
    return trainer


def test_callback_uploads_periodically_and_on_finish(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_fd_backed_std_streams(monkeypatch)
    trainer = _mlflow_trainer()
    client = trainer.logger.experiment
    callback = LogConsoleToMLflowCallback(upload_interval_seconds=0.05)

    callback.setup(trainer, MagicMock(), stage="fit")
    try:
        print("hello from training")
        deadline = time.monotonic() + 5
        while not client.log_artifact.called and time.monotonic() < deadline:
            time.sleep(0.01)
        assert client.log_artifact.called, "no periodic upload happened"
    finally:
        callback.finish()

    run_id, path = client.log_artifact.call_args.args
    assert run_id == "run-123"
    assert client.log_artifact.call_args.kwargs == {"artifact_path": "logs"}
    assert Path(path).name == "console.log"
    assert "hello from training" in Path(path).read_text()

    n_calls = client.log_artifact.call_count
    callback.finish()  # idempotent: no second final upload
    assert client.log_artifact.call_count == n_calls


def test_callback_upload_failure_does_not_raise() -> None:
    trainer = _mlflow_trainer()
    trainer.logger.experiment.log_artifact.side_effect = OSError("server down")
    callback = LogConsoleToMLflowCallback(upload_interval_seconds=0.05)

    callback.setup(trainer, MagicMock(), stage="fit")
    time.sleep(0.2)
    callback.finish()

    assert trainer.logger.experiment.log_artifact.call_count >= 2


@pytest.mark.parametrize("is_global_zero", [False, True])
def test_callback_noop_off_rank_zero_or_without_mlflow(is_global_zero: bool) -> None:
    trainer = _mlflow_trainer(is_global_zero=is_global_zero)
    if is_global_zero:
        trainer.logger = MagicMock()  # not an MLFlowLogger
    callback = LogConsoleToMLflowCallback()

    callback.setup(trainer, MagicMock(), stage="fit")

    assert callback._capture is None
    callback.finish()
