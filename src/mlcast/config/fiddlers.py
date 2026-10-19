"""Fiddler mutators for high-level semantic configuration changes.

Fiddlers are functions that accept a ``fdl.Config`` and mutate it in place.
They are the right tool when a change spans multiple config parameters that
must stay in sync — for example, switching the dataset class while preserving
its existing parameters, or enabling masking consistently across both the data
pipeline and the loss function.

Use fiddlers from the CLI via ``--config fiddler:<name>`` or
``--config "fiddler:<name>(arg=value)"``, or call them directly on a buildable
config in Python before passing it to ``fdl.build()``.
"""

import inspect
import os

import fiddle as fdl
from loguru import logger
from pytorch_lightning.callbacks import ProgressBar, TQDMProgressBar
from pytorch_lightning.loggers import MLFlowLogger

from ..callbacks import LogConsoleToMLflowCallback, LogSystemInfoCallback, LogTrainingProgressCallback
from ..data.source_data_datasets import SourceDataRandomSamplingDataset
from ..loggers import ResilientMLFlowLogger


def set_variables(cfg: fdl.Config, standard_names: list[str]) -> None:
    """Fiddler to synchronize dataset variables with the network's input channels.

    Sets ``dataset_factory.standard_names`` on the data config and, when the
    network config exposes an ``input_channels`` parameter (e.g.
    ``ConvGruModel``), keeps it in sync.  Networks that use a different
    parameter name for the channel count (e.g. ``HalfUNet`` uses
    ``in_channels``) are left unchanged — callers are responsible for keeping
    that parameter consistent when swapping in an external architecture.

    Parameters
    ----------
    cfg : fdl.Config
        The Fiddle configuration to mutate.
    standard_names : list of str
        The new list of standard names to load.
    """
    cfg.data.dataset_factory.standard_names = standard_names
    network_cls = cfg.pl_module.network.__fn_or_cls__
    sig = inspect.signature(network_cls.__init__)
    if "input_channels" in sig.parameters:
        cfg.pl_module.network.input_channels = len(standard_names)
    else:
        logger.warning(
            "set_variables: network {} has no 'input_channels' parameter; "
            "channel count not updated. Set it manually on the network config.",
            network_cls.__name__,
        )


def toggle_masking(cfg: fdl.Config, enabled: bool) -> None:
    """Fiddler to synchronize dataset mask yielding with masked loss computation.

    Parameters
    ----------
    cfg : fdl.Config
        The Fiddle configuration to mutate.
    enabled : bool
        Whether to enable masking or not.
    """
    cfg.data.dataset_factory.return_mask = enabled
    cfg.pl_module.masked_loss = enabled


def use_random_sampler(cfg: fdl.Config) -> None:
    """Fiddler to switch the dataset factory to use the random sampler.

    Parameters
    ----------
    cfg : fdl.Config
        The Fiddle configuration to mutate.
    """
    # Keep the existing parameters but change the underlying class
    cfg.data.dataset_factory = fdl.Partial(
        SourceDataRandomSamplingDataset,
        zarr_path=cfg.data.dataset_factory.zarr_path,
        standard_names=cfg.data.dataset_factory.standard_names,
        input_steps=cfg.data.dataset_factory.input_steps,
        forecast_steps=cfg.data.dataset_factory.forecast_steps,
        return_mask=cfg.data.dataset_factory.return_mask,
        storage_options=getattr(cfg.data.dataset_factory, "storage_options", None),
    )


def use_ratio_splits(cfg: fdl.Config, train: float, val: float) -> None:
    """Fiddler to set fraction-based train/val/test splits on the data module."""
    cfg.data.splits = {"time": {"train": train, "val": val, "test": 1.0 - train - val}}


def use_anon_s3_dataset(cfg: fdl.Buildable, zarr_path: str, endpoint_url: str) -> None:
    """Configure the dataset factory to read anonymously from an S3 object store.

    Parameters
    ----------
    cfg : fdl.Buildable
        The Fiddle configuration to mutate.
    zarr_path : str
        The S3 URI path to the Zarr dataset (e.g., s3://bucket/path.zarr).
    endpoint_url : str
        The endpoint URL for the S3 object store.
    """
    cfg.data.dataset_factory.zarr_path = zarr_path
    cfg.data.dataset_factory.storage_options = {
        "anon": True,
        "client_kwargs": {
            "endpoint_url": endpoint_url,
            "verify": False,
        },
        "config_kwargs": {"signature_version": "s3v4"},
    }


def use_mlflow_logger(
    cfg: fdl.Config,
    tracking_uri: str | None = None,
    console_upload_interval_seconds: float = 60.0,
    progress_log_interval_seconds: float = 60.0,
    upload_checkpoints: bool = False,
) -> None:
    """Fiddler to switch the trainer logger to MLflow.

    Replaces the default TensorBoardLogger with an MLFlowLogger
    (`ResilientMLFlowLogger`), inheriting
    the experiment name from the existing logger config. The run name is
    left unset, deferring to the ``MLFLOW_RUN_NAME`` environment variable
    (or MLflow's default). Also adds callbacks that log system info and
    metrics, that upload the console output to the run as
    ``logs/console.log`` while training, and that log a progress line at a
    fixed interval so progress shows in that log. Unless a progress bar is
    already configured, the progress bar is set to TQDM, because the console
    capture hides Lightning's default Rich bar.

    Parameters
    ----------
    cfg : fdl.Config
        The Fiddle configuration to mutate.
    tracking_uri : str, optional
        Explicit MLflow tracking server URI. When omitted, defers to the
        ``MLFLOW_TRACKING_URI`` environment variable (or MLflow's local
        ``./mlruns`` default) — and a missing env var still warns in that
        case, same as before.
    console_upload_interval_seconds : float, optional
        How often the captured console log is re-uploaded to MLflow while
        training runs. Default 60.
    progress_log_interval_seconds : float, optional
        Minimum seconds between the progress lines logged during training.
        Default 60.
    upload_checkpoints : bool, optional
        Upload every checkpoint to the run as it is saved, so the run can be
        resumed from another machine with ``--config mlflow:<run>``. Off by
        default: checkpoints are large (~770 MB for the 5-block ConvGRU) and
        some servers refuse them (mlflow.dmidev.org does). A failed upload
        is logged as a warning and doesn't stop training, but each attempt
        can block it for minutes of retries.
    """
    if tracking_uri is None and not os.environ.get("MLFLOW_TRACKING_URI"):
        logger.warning(
            "MLFLOW_TRACKING_URI is not set. MLflow will log to a local './mlruns' directory. "
            "Set MLFLOW_TRACKING_URI to point to a remote tracking server, "
            "e.g. export MLFLOW_TRACKING_URI=http://localhost:5000"
        )
    # log_model="all" uploads every checkpoint to the run as it is saved.
    cfg.trainer.logger = fdl.Config(
        ResilientMLFlowLogger,
        experiment_name=cfg.trainer.logger.name,
        log_model="all" if upload_checkpoints else False,
    )
    if tracking_uri is not None:
        cfg.trainer.logger.tracking_uri = tracking_uri
    cfg.trainer.callbacks.append(fdl.Config(LogSystemInfoCallback))
    cfg.trainer.callbacks.append(
        fdl.Config(LogConsoleToMLflowCallback, upload_interval_seconds=console_upload_interval_seconds)
    )
    cfg.trainer.callbacks.append(
        fdl.Config(LogTrainingProgressCallback, log_interval_seconds=progress_log_interval_seconds)
    )
    # The console capture points fds 1/2 at pipes, so Lightning's default
    # RichProgressBar sees no terminal and hides its live display. TQDM redraws
    # with "\r" regardless, which passes through to the terminal (and which the
    # capture collapses in the log file). Lightning raises on two progress
    # bars, so leave an explicitly configured one alone. Callback entries are
    # `fdl.Config` nodes here, hence the check on `__fn_or_cls__`.
    has_progress_bar = any(
        isinstance(getattr(cb, "__fn_or_cls__", None), type) and issubclass(cb.__fn_or_cls__, ProgressBar)
        for cb in cfg.trainer.callbacks
    )
    if not has_progress_bar:
        cfg.trainer.callbacks.append(fdl.Config(TQDMProgressBar))


def resume_training(
    cfg: fdl.Config,
    ckpt_path: str,
    mlflow_run_id: str | None = None,
    inplace: bool = False,
    run_name: str | None = None,
) -> None:
    """Fiddler to resume training from a checkpoint.

    Sets ``cfg.ckpt_path`` so that ``Trainer.fit`` restores the full
    training state (weights, optimizer, LR scheduler, step counter and the
    ``max_time`` timer, so ``max_time`` is the total budget, not the extra
    time). ``mlcast train --config mlflow:<run>`` applies this fiddler for you.

    With ``mlflow_run_id`` (the run being continued), the continuation is
    logged as a **new MLflow run linked to it**: tagged
    ``mlflow.parentRunId`` (so the UI nests it under the original),
    ``mlcast.resumed_from_run`` and ``mlcast.resumed_from_checkpoint``. A new
    run can log its own config and status, which MLflow wouldn't allow in
    the original run (params are immutable). Step numbers continue from the
    checkpoint, so the runs' curves line up. With ``inplace=True`` the
    original run is continued instead. Checkpoint upload stays as the
    config has it (``set:trainer.logger.log_model=...`` to change it), and
    the logger is switched to `ResilientMLFlowLogger`, so configs saved by
    older runs with plain ``MLFlowLogger`` can't crash on an upload either.

    Parameters
    ----------
    cfg : fdl.Config
        The Fiddle configuration to mutate.
    ckpt_path : str
        Local path of the checkpoint to resume from.
    mlflow_run_id : str, optional
        MLflow run being continued. Requires the trainer logger to be an
        ``MLFlowLogger``.
    inplace : bool, optional
        Log into ``mlflow_run_id`` itself instead of a new linked run.
    run_name : str, optional
        Name for the new run (ignored with ``inplace``).
    """
    cfg.ckpt_path = ckpt_path
    if mlflow_run_id is None:
        return
    logger_cls = getattr(cfg.trainer.logger, "__fn_or_cls__", None)
    if not (isinstance(logger_cls, type) and issubclass(logger_cls, MLFlowLogger)):
        raise ValueError("resume_training(mlflow_run_id=...) needs an MLFlowLogger; apply use_mlflow_logger first")
    fdl.update_callable(cfg.trainer.logger, ResilientMLFlowLogger)
    if inplace:
        cfg.trainer.logger.run_id = mlflow_run_id
        return
    # A config saved by an earlier (in-place or linked) resume may carry that
    # run's id, name and links; a new run must not inherit them.
    cfg.trainer.logger.run_id = None
    cfg.trainer.logger.run_name = run_name
    cfg.trainer.logger.tags = {
        **(cfg.trainer.logger.tags or {}),
        "mlflow.parentRunId": mlflow_run_id,
        "mlcast.resumed_from_run": mlflow_run_id,
        "mlcast.resumed_from_checkpoint": os.path.basename(ckpt_path),
    }
