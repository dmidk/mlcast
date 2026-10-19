"""Lightning loggers for mlcast."""

from loguru import logger as log
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.loggers.utilities import _scan_checkpoints


class ResilientMLFlowLogger(MLFlowLogger):
    """``MLFlowLogger`` whose checkpoint uploads can't take down a training run.

    With ``log_model`` set, Lightning's ``MLFlowLogger`` uploads checkpoints
    inside the training loop and lets any upload error propagate, which stops
    training. Checkpoints are large (about 770 MB for the 5-block ConvGRU),
    and a tracking server or its proxy may refuse them: mlflow.dmidev.org cut
    the connection on every attempt and a 22 h run died at its first
    validation. Here a failed upload is logged as a warning instead, and the
    checkpoints it covered are marked as handled so that later saves don't
    retry them (each attempt can block training for minutes of HTTP
    retries). The checkpoint files themselves are written locally before any
    upload, so nothing is lost but the copy on the server.
    """

    def _scan_and_log_checkpoints(self, checkpoint_callback: ModelCheckpoint) -> None:
        # Overrides a private Lightning method, the one both the per-save
        # (`after_save_checkpoint`) and end-of-training (`finalize`) uploads go through.
        try:
            super()._scan_and_log_checkpoints(checkpoint_callback)
        except Exception as e:  # noqa: BLE001
            pending = _scan_checkpoints(checkpoint_callback, self._logged_model_time)
            for timestamp, path, _, _ in pending:
                self._logged_model_time[path] = timestamp
            names = ", ".join(str(p) for _, p, _, _ in pending) or "checkpoint"
            log.warning(
                f"Could not upload {names} to MLflow; training continues and the checkpoint stays on local disk. "
                f"To stop trying, don't enable checkpoint upload. Error: {type(e).__name__}: {e}"[:1000]
            )
