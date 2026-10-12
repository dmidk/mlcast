"""PyTorch Lightning callbacks for mlcast."""

import atexit
import codecs
import os
import platform
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import psutil
import pytorch_lightning as pl
import torch
from loguru import logger as log
from mlflow.system_metrics.system_metrics_monitor import SystemMetricsMonitor
from pytorch_lightning.loggers import MLFlowLogger


def _get_system_tags() -> dict[str, str]:
    """Collect system and environment metadata as a flat string dict."""
    tags = {
        "sys.hostname": platform.node(),
        "sys.os": platform.platform(),
        "sys.python_version": f"{platform.python_implementation()} {sys.version.split()[0]}",
        "sys.python_executable": sys.executable,
        "sys.command": " ".join(sys.argv),
        "sys.cpu_count": str(psutil.cpu_count(logical=False)),
        "sys.logical_cpu_count": str(psutil.cpu_count(logical=True)),
        "sys.gpu_count": str(torch.cuda.device_count()),
    }

    for i in range(torch.cuda.device_count()):
        tags[f"sys.gpu_{i}_type"] = torch.cuda.get_device_name(i)

    try:
        tags["sys.git_repository"] = subprocess.check_output(
            ["git", "remote", "get-url", "origin"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        tags["sys.git_commit"] = commit
        tags["sys.git_branch"] = branch
        tags["sys.git_state"] = f'git checkout -b "{branch}" {commit}'
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return tags


class LogSystemInfoCallback(pl.Callback):
    """Logs system and environment metadata as MLflow run tags at training start.

    Collects hostname, OS, Python version, git state, CPU/GPU info and logs
    them as tags on the active MLflow run. Also starts MLflow's
    SystemMetricsMonitor to log CPU/GPU/memory metrics throughout training.
    No-ops silently for other loggers.
    """

    def __init__(self) -> None:
        self._system_monitor: SystemMetricsMonitor | None = None

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not isinstance(trainer.logger, MLFlowLogger):
            return

        tags = _get_system_tags()
        for key, value in tags.items():
            trainer.logger.experiment.set_tag(trainer.logger.run_id, key, value)

        tracking_uri = trainer.logger._tracking_uri
        experiment_id = trainer.logger._experiment_id
        run_id = trainer.logger.run_id

        self._system_monitor = SystemMetricsMonitor(run_id=run_id, tracking_uri=tracking_uri)
        self._system_monitor.start()

        run_url = f"{tracking_uri}/#/experiments/{experiment_id}/runs/{run_id}"
        log.info(f"MLflow run started: {run_url}")

    def _stop_monitor(self) -> None:
        if self._system_monitor is not None:
            self._system_monitor.finish()
            self._system_monitor = None

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._stop_monitor()

    def on_exception(self, trainer: pl.Trainer, pl_module: pl.LightningModule, exception: BaseException) -> None:
        self._stop_monitor()


def _write_all(fd: int, data: bytes) -> None:
    """``os.write`` until all of `data` is written (pipes may accept partial writes)."""
    while data:
        data = data[os.write(fd, data) :]


def _collapse_redraws(line: str) -> str:
    """Keep only what a terminal would show after carriage-return redraws.

    Progress bars redraw a line by writing ``\\r`` followed by the new state.
    The text after the last ``\\r`` is kept; a trailing ``\\r`` is preserved
    so the next write is still known to overwrite the line.
    """
    body = line.rstrip("\r")
    return body[body.rfind("\r") + 1 :] + line[len(body) : len(body) + 1]


class _ConsoleCapture:
    """Copy everything written to this process's stdout and stderr into a file.

    Works at the file-descriptor level (fds 1 and 2), so it also captures
    output that doesn't go through the current ``sys.stdout``/``sys.stderr``
    objects: loguru and stdlib logging handlers bound to the original
    streams, C extensions, and the interpreter's traceback for an uncaught
    exception. Output is still passed through to the original streams.
    Progress-bar redraws are collapsed so that only the final state of each
    line is written to the file.

    Parameters
    ----------
    path : Path
        File to write the captured output to.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._file = path.open("w", encoding="utf-8")
        self._file_lock = threading.Lock()
        self._saved_fds: dict[int, int] = {}
        self._pumps: list[threading.Thread] = []

    def start(self) -> None:
        sys.stdout.flush()
        sys.stderr.flush()
        for fd in (1, 2):
            self._saved_fds[fd] = os.dup(fd)
            read_fd, write_fd = os.pipe()
            os.dup2(write_fd, fd)
            os.close(write_fd)
            pump = threading.Thread(
                target=self._pump, args=(read_fd, self._saved_fds[fd]), daemon=True, name=f"console-capture-fd{fd}"
            )
            pump.start()
            self._pumps.append(pump)

    def _pump(self, read_fd: int, passthrough_fd: int) -> None:
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        line = ""
        passthrough_ok = True
        while chunk := os.read(read_fd, 65536):
            if passthrough_ok:
                try:
                    _write_all(passthrough_fd, chunk)
                except OSError:
                    # The original stream is gone (e.g. the terminal of a
                    # nohup'ed run disconnected); keep capturing to the file.
                    passthrough_ok = False
            *complete, line = (line + decoder.decode(chunk)).split("\n")
            line = _collapse_redraws(line)
            if complete:
                self._write_lines([_collapse_redraws(c).rstrip("\r") for c in complete])
        os.close(read_fd)
        if line.rstrip("\r"):
            self._write_lines([line.rstrip("\r")])

    def _write_lines(self, lines: list[str]) -> None:
        with self._file_lock:
            if not self._file.closed:
                self._file.write("\n".join(lines) + "\n")
                self._file.flush()

    def stop(self, timeout: float = 2.0) -> None:
        """Restore the original stdout/stderr and finish writing the file.

        Restoring the fds closes this process's write end of each pipe, so
        the pumps see EOF once everything written so far has been drained.
        Child processes that inherited the pipe (e.g. dataloader workers
        still alive at exit) keep it open, hence the bounded wait.
        """
        sys.stdout.flush()
        sys.stderr.flush()
        for fd, saved in self._saved_fds.items():
            os.dup2(saved, fd)
        for pump in self._pumps:
            pump.join(timeout)
        with self._file_lock:
            self._file.close()


class LogConsoleToMLflowCallback(pl.Callback):
    """Upload the training process's console output to MLflow as ``logs/console.log``.

    From trainer setup onwards, everything written to stdout and stderr is
    copied to a local file (see `_ConsoleCapture`), which a background thread
    re-uploads to the run every `upload_interval_seconds`, so the log can be
    followed in the MLflow UI while training runs. Only the global-zero
    process captures and uploads. No-ops for other loggers.

    The final upload happens at interpreter exit rather than in a trainer
    hook: when training crashes, Lightning calls ``on_exception`` and then
    re-raises, and the traceback is only printed once the exception reaches
    the top level, just before ``atexit`` handlers run.

    Parameters
    ----------
    upload_interval_seconds : float
        Seconds between uploads while training runs.
    """

    def __init__(self, upload_interval_seconds: float = 60.0) -> None:
        self.upload_interval_seconds = upload_interval_seconds
        self._capture: _ConsoleCapture | None = None
        self._uploader: threading.Thread | None = None
        self._stop_uploading = threading.Event()
        self._client = None
        self._run_id: str | None = None
        self._upload_failing = False

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        if self._capture is not None or not trainer.is_global_zero or not isinstance(trainer.logger, MLFlowLogger):
            return
        self._client = trainer.logger.experiment
        self._run_id = trainer.logger.run_id
        self._capture = _ConsoleCapture(Path(tempfile.mkdtemp(prefix="mlcast-console-")) / "console.log")
        self._capture.start()
        self._stop_uploading.clear()
        self._uploader = threading.Thread(target=self._upload_periodically, daemon=True, name="console-log-upload")
        self._uploader.start()
        atexit.register(self.finish)
        log.info(f"Uploading console output to MLflow every {self.upload_interval_seconds:g}s as logs/console.log")

    def _upload_periodically(self) -> None:
        while not self._stop_uploading.wait(self.upload_interval_seconds):
            self._upload()

    def _upload(self) -> None:
        try:
            self._client.log_artifact(self._run_id, str(self._capture.path), artifact_path="logs")
        except Exception as e:  # noqa: BLE001
            # Never let a log upload interrupt training; warn once per outage.
            if not self._upload_failing:
                log.warning(f"Could not upload console log to MLflow (will keep retrying): {e!r}")
            self._upload_failing = True
        else:
            self._upload_failing = False

    def finish(self) -> None:
        """Stop capturing and do a final upload. Safe to call more than once."""
        if self._capture is None:
            return
        self._stop_uploading.set()
        self._uploader.join()
        self._capture.stop()
        self._upload()
        self._capture = None
        atexit.unregister(self.finish)
