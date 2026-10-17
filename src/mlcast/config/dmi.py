"""DMI 500m/10min radar composite (v012) ConvGRU training config.

Derives from the default `training_experiment` config graph (same model,
optimizer, and trainer defaults), swapping in DMI-specific dataset paths,
forecast horizon, and fixed calendar-year splits instead of the default
ratio split.
"""

import datetime

import fiddle as fdl
from pytorch_lightning.callbacks import ModelCheckpoint

from .base import Experiment, training_experiment
from .fiddlers import set_variables, use_mlflow_logger

ZARR_PATH = "/dmidata/projects/radar/products/composite/zarrComposite/tmp_folder/DMI_500m_10min_v012.zarr"
INDEX_PATH = (
    "/dmidata/projects/cloudphysics/mlcast/sampling_index_2016-02-29-2025-12-31_24x256x256_3x16x16_10000.parquet"
)
MLFLOW_TRACKING_URI = "https://mlflow.dmidev.org/"
CHECKPOINT_EVERY_HOURS = 3


def dmi_500m_10min_convgru_experiment() -> fdl.Config[Experiment]:
    """Build a Fiddle config for ConvGRU training on DMI's 500m/10min composite.

    Starts from `training_experiment.as_buildable()` (same model/optimizer/
    trainer defaults) and overrides:

    - the dataset factory's `zarr_path`/`index_path` to point at the DMI
      composite and its precomputed sampling index;
    - `input_steps`/`forecast_steps` to 6/18 (1h in -> 3h lead), summing to
      exactly the sampling index's 24-frame/10-min datacube depth -- see
      the index filename's `{time_depth}x{width}x{height}` field;
    - `standard_names` to `["equivalent_reflectivity_factor"]`, via the
      `set_variables` fiddler so the network's `input_channels` stays in
      sync;
    - `batch_size` to 32 (default 16 left the A40 on ohm.dmi.dk at ~43%
      memory utilization);
    - `data.splits` to fixed date ranges (2019-07-13 to 2023 train, since
      earlier data is missing / 2024 val / 2025 test) instead of the
      default ratio split, so the 2025
      test season can never be contaminated by a ratio boundary landing
      mid-year;
    - an extra checkpoint callback saving a checkpoint every
      `CHECKPOINT_EVERY_HOURS` hours of training, kept alongside the
      best-loss checkpoints;
    - the trainer logger to MLflow, pointed at the project's tracking
      server.

    Not `@auto_config`-decorated, deliberately: that decorator only
    supports constructing a graph fresh (as if eager), not mutating an
    existing one -- a multi-level attribute-assignment target off a
    dataclass-typed config (e.g. `cfg.data.dataset_factory.zarr_path = ...`
    where `cfg` is an inlined `training_experiment()` call) raises inside
    an `@auto_config` body, and routing it through a plain helper function
    doesn't help either: calls to non-`@auto_config` functions from inside
    an `@auto_config` body never execute -- they get wrapped as inert
    config nodes, the same mechanism that turns `ConvGruModel(...)` into
    `fdl.Config(ConvGruModel, ...)`. A plain function that returns a
    `fdl.Config` directly works identically from the CLI --
    `fiddle.absl_flags` calls a resolved target directly whenever it isn't
    an `AutoConfig` instance -- and for `--config set:...` overrides on
    top.

    Call `mlcast train --config=config:dmi_500m_10min_convgru_experiment`,
    optionally with further `--config set:...` overrides on top, exactly
    as for the default config.

    Returns
    -------
    fdl.Config[Experiment]
        Buildable experiment config with model, data, and trainer.
    """
    cfg = training_experiment.as_buildable()

    cfg.data.dataset_factory.zarr_path = ZARR_PATH
    cfg.data.dataset_factory.index_path = INDEX_PATH
    cfg.data.dataset_factory.input_steps = 6
    cfg.data.dataset_factory.forecast_steps = 18
    set_variables(cfg, standard_names=["equivalent_reflectivity_factor"])

    # Default batch_size=16 left the A40 on ohm.dmi.dk at ~43% memory
    # utilization; double it rather than leaving headroom unused.
    cfg.data.batch_size = 32

    # Tuple-range mode (see mlcast.data.splits): explicit inclusive date
    # ranges, not mlcast's default fraction-of-timeline split.
    cfg.data.splits = {
        "time": {
            "train": ("2019-07-13", "2023-12-31"),
            "val": ("2024-01-01", "2024-12-31"),
            "test": ("2025-01-01", "2025-12-31"),
        }
    }

    # Time-based checkpoints (Lightning's ModelCheckpoint `train_time_interval`,
    # checked at the end of each training batch), kept alongside the
    # best-val_loss/train_loss checkpoints. `save_top_k=-1` keeps every one; the
    # distinct filename prefix stops them colliding with the top-k callbacks'
    # default `{epoch}-{step}` names. The timedelta is itself an fdl.Config so
    # the config graph stays serialisable to YAML.
    cfg.trainer.callbacks.append(
        fdl.Config(
            ModelCheckpoint,
            filename=f"every-{CHECKPOINT_EVERY_HOURS}h-{{epoch}}-{{step}}",
            save_top_k=-1,
            train_time_interval=fdl.Config(datetime.timedelta, hours=CHECKPOINT_EVERY_HOURS),
        )
    )

    cfg.trainer.logger.name = "dmi-500m-10min-convgru"
    use_mlflow_logger(cfg, tracking_uri=MLFLOW_TRACKING_URI)

    return cfg
