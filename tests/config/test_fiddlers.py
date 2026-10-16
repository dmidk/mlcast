import fiddle as fdl
from pytorch_lightning.callbacks import ProgressBar, RichProgressBar, TQDMProgressBar
from pytorch_lightning.loggers import MLFlowLogger

from mlcast.callbacks import LogConsoleToMLflowCallback, LogTrainingProgressCallback
from mlcast.config import set_variables, toggle_masking, training_experiment, use_mlflow_logger


def test_fiddler_set_variables():
    """Verify set_variables syncs dataset variables and network input_channels."""
    cfg = training_experiment.as_buildable()

    # Apply fiddler
    set_variables(cfg, ["rainfall_rate", "rainfall_flux"])

    # Check sync
    assert cfg.data.dataset_factory.standard_names == ["rainfall_rate", "rainfall_flux"]
    assert cfg.pl_module.network.input_channels == 2


def test_fiddler_toggle_masking():
    """Verify toggle_masking syncs dataset mask return and module masked_loss."""
    cfg = training_experiment.as_buildable()

    # Disable masking
    toggle_masking(cfg, False)
    assert cfg.data.dataset_factory.return_mask is False
    assert cfg.pl_module.masked_loss is False

    # Enable masking
    toggle_masking(cfg, True)
    assert cfg.data.dataset_factory.return_mask is True
    assert cfg.pl_module.masked_loss is True


def test_fiddler_use_mlflow_logger_explicit_tracking_uri(monkeypatch):
    """An explicit tracking_uri is applied without needing MLFLOW_TRACKING_URI set."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    cfg = training_experiment.as_buildable()

    use_mlflow_logger(cfg, tracking_uri="https://mlflow.example.org/")

    assert cfg.trainer.logger.__fn_or_cls__ is MLFlowLogger
    assert cfg.trainer.logger.tracking_uri == "https://mlflow.example.org/"


def test_fiddler_use_mlflow_logger_defers_to_env_var(monkeypatch):
    """With no explicit tracking_uri, the logger config carries no tracking_uri of its own."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    cfg = training_experiment.as_buildable()

    use_mlflow_logger(cfg)

    assert cfg.trainer.logger.__fn_or_cls__ is MLFlowLogger
    assert not hasattr(cfg.trainer.logger, "tracking_uri") or cfg.trainer.logger.tracking_uri is None


def test_fiddler_use_mlflow_logger_adds_console_log_callback(monkeypatch):
    """The console-log upload callback is added, with the interval passed through."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.org/")
    cfg = training_experiment.as_buildable()

    use_mlflow_logger(cfg, console_upload_interval_seconds=15.0)

    (console_cb,) = [cb for cb in cfg.trainer.callbacks if cb.__fn_or_cls__ is LogConsoleToMLflowCallback]
    assert console_cb.upload_interval_seconds == 15.0


def test_fiddler_use_mlflow_logger_adds_progress_logging_and_tqdm_bar(monkeypatch):
    """The progress-line callback and a TQDM progress bar are added once each."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.org/")
    cfg = training_experiment.as_buildable()

    use_mlflow_logger(cfg, progress_log_interval_seconds=5.0)

    (progress_cb,) = [cb for cb in cfg.trainer.callbacks if cb.__fn_or_cls__ is LogTrainingProgressCallback]
    assert progress_cb.log_interval_seconds == 5.0
    assert [cb.__fn_or_cls__ for cb in cfg.trainer.callbacks].count(TQDMProgressBar) == 1


def test_fiddler_use_mlflow_logger_keeps_configured_progress_bar(monkeypatch):
    """A progress bar that is already configured is left alone, not doubled."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.org/")
    cfg = training_experiment.as_buildable()
    cfg.trainer.callbacks.append(fdl.Config(RichProgressBar))

    use_mlflow_logger(cfg)

    bars = [cb.__fn_or_cls__ for cb in cfg.trainer.callbacks if issubclass(cb.__fn_or_cls__, ProgressBar)]
    assert bars == [RichProgressBar]
