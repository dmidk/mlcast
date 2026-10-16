"""Tests for the console-log capture, its MLflow upload callback and the progress-line callback."""

from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from loguru import logger as log
from pytorch_lightning.callbacks import Timer
from pytorch_lightning.loggers import MLFlowLogger

import mlcast.callbacks as callbacks_module
from mlcast.callbacks import (
    LogConsoleToMLflowCallback,
    LogTrainingProgressCallback,
    _collapse_redraws,
    _ConsoleCapture,
)


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


class _FakeClock:
    """Stand-in for `time.monotonic` that only moves when told to."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _progress_trainer(is_global_zero: bool = True) -> MagicMock:
    trainer = MagicMock()
    trainer.is_global_zero = is_global_zero
    trainer.global_step = 0
    trainer.current_epoch = 0
    trainer.num_training_batches = 1000
    trainer.callback_metrics = {"train_loss_step": torch.tensor(0.25)}
    timer = MagicMock(spec=Timer)
    timer.time_remaining.return_value = 3600.0
    trainer.callbacks = [timer]
    return trainer


def _capture_log_messages() -> tuple[list[str], int]:
    messages: list[str] = []
    sink_id = log.add(lambda m: messages.append(m.record["message"]), level="INFO")
    return messages, sink_id


def test_progress_callback_logs_once_per_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _FakeClock()
    monkeypatch.setattr(callbacks_module.time, "monotonic", clock)
    trainer = _progress_trainer()
    callback = LogTrainingProgressCallback(log_interval_seconds=60.0)
    batch = {"input": torch.zeros(16, 6, 1, 4, 4)}
    messages, sink_id = _capture_log_messages()
    try:
        callback.on_train_start(trainer, MagicMock())
        # 30 batches over 30 s: below the interval, nothing logged.
        for i in range(30):
            clock.now += 1.0
            trainer.global_step = i + 1
            callback.on_train_batch_end(trainer, MagicMock(), None, batch, i)
        assert not [m for m in messages if m.startswith("Training progress")]
        # Another 30 batches reach the 60 s interval: exactly one line.
        for i in range(30, 60):
            clock.now += 1.0
            trainer.global_step = i + 1
            callback.on_train_batch_end(trainer, MagicMock(), None, batch, i)
    finally:
        log.remove(sink_id)

    (line,) = [m for m in messages if m.startswith("Training progress")]
    assert "step 60" in line
    assert "epoch 0 batch 60/1000" in line
    assert "train_loss_step 0.25" in line
    assert "1 steps/s" in line
    assert "16 samples/s" in line
    assert "elapsed 0:01:00" in line
    assert "remaining 1:00:00" in line


def test_progress_callback_noop_off_rank_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _FakeClock()
    monkeypatch.setattr(callbacks_module.time, "monotonic", clock)
    trainer = _progress_trainer(is_global_zero=False)
    callback = LogTrainingProgressCallback(log_interval_seconds=1.0)
    messages, sink_id = _capture_log_messages()
    try:
        callback.on_train_start(trainer, MagicMock())
        clock.now += 10.0
        callback.on_train_batch_end(trainer, MagicMock(), None, {"input": torch.zeros(2, 1)}, 0)
    finally:
        log.remove(sink_id)

    assert not [m for m in messages if m.startswith("Training progress")]
