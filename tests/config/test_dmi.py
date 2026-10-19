import datetime
from pathlib import Path

import fiddle as fdl
from fiddle.experimental.yaml_serialization import dump_yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from mlcast.config import dmi_500m_10min_convgru_experiment, load_yaml_config, validate_config


def test_dmi_500m_10min_convgru_experiment_config():
    cfg = dmi_500m_10min_convgru_experiment()  # plain function, already returns fdl.Config

    assert cfg.data.dataset_factory.zarr_path.endswith("DMI_500m_10min_v012.zarr")
    assert cfg.data.dataset_factory.index_path.endswith(".parquet")
    assert cfg.data.dataset_factory.input_steps == 6
    assert cfg.data.dataset_factory.forecast_steps == 18
    assert cfg.data.dataset_factory.standard_names == ["equivalent_reflectivity_factor"]
    assert cfg.pl_module.network.input_channels == 1
    assert cfg.data.batch_size == 32

    assert cfg.data.splits == {
        "time": {
            "train": ("2019-07-13", "2023-12-31"),
            "val": ("2024-01-01", "2024-12-31"),
            "test": ("2025-01-01", "2025-12-31"),
        }
    }

    assert issubclass(cfg.trainer.logger.__fn_or_cls__, MLFlowLogger)
    assert cfg.trainer.logger.log_model is False  # mlflow.dmidev.org refuses ~770 MB checkpoint uploads
    assert cfg.trainer.logger.tracking_uri == "https://mlflow.dmidev.org/"
    assert cfg.trainer.logger.experiment_name == "dmi-500m-10min-convgru"

    validate_config(cfg)
    fdl.build(cfg)  # succeeds without touching the filesystem


def _time_based_checkpoints(callbacks: list) -> list[ModelCheckpoint]:
    # `_train_time_interval` is Lightning's private attribute holding the
    # `train_time_interval` argument; ModelCheckpoint has no public accessor.
    return [cb for cb in callbacks if isinstance(cb, ModelCheckpoint) and cb._train_time_interval is not None]


def test_dmi_config_checkpoints_every_3_hours(tmp_path: Path) -> None:
    cfg = dmi_500m_10min_convgru_experiment()

    (ckpt,) = _time_based_checkpoints(fdl.build(cfg).trainer.callbacks)
    assert ckpt._train_time_interval == datetime.timedelta(hours=3)
    assert ckpt.save_top_k == -1
    assert ckpt.filename == "every-3h-{epoch}-{step}"

    # The timedelta must survive the YAML config that's saved with every run.
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(dump_yaml(cfg))
    reloaded = load_yaml_config(str(yaml_path))
    (reloaded_ckpt,) = _time_based_checkpoints([fdl.build(cb) for cb in reloaded.trainer.callbacks])
    assert reloaded_ckpt._train_time_interval == datetime.timedelta(hours=3)


def test_dmi_config_builds_after_yaml_round_trip(tmp_path: Path) -> None:
    """The saved config.yaml (what `--config config.yaml` and `--config mlflow:<run>` load)
    builds again. YAML has no tuples, so the (start, end) date-range splits come back as
    lists; the data module turns them back into tuples."""
    cfg = dmi_500m_10min_convgru_experiment()
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(dump_yaml(cfg))

    experiment = fdl.build(load_yaml_config(str(yaml_path)))

    assert experiment.data.splits == {
        "time": {
            "train": ("2019-07-13", "2023-12-31"),
            "val": ("2024-01-01", "2024-12-31"),
            "test": ("2025-01-01", "2025-12-31"),
        }
    }
