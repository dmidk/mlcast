# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Dataset sampling folded into `mlcast` from the standalone `mlcast-dataset-sampler` as
  `mlcast.data.source_data.sampling`. `mlcast build-sampling-index` scans a source Zarr (on CPU, or GPU with
  `--device cuda`) and writes a parquet sampling index of candidate datacubes with per-datacube stats
  (`nan_count`, `sum`, `mean`, `frac_wet`) and the sampling parameters in its metadata;
  `mlcast validate-sampling-index` checks a file against that contract.
  [#17](https://github.com/mlcast-community/mlcast/pull/17) @franchg @leifdenby @noralinn
- Pluggable candidate selectors (`CandidateSelector`, `ImportanceSelector`, `UniformSelector`, with a
  `register_selector`/`get_selector` registry) that pick the subset of the index to train on, once at dataset
  init. `SourceDataDataModule` takes per-split `train_selector` and `eval_selector` arguments; the default
  experiment uses importance sampling on train and uniform 10% sampling on val/test.
  [#17](https://github.com/mlcast-community/mlcast/pull/17) @franchg @leifdenby @noralinn
- (pending upstream) With `fiddler:use_mlflow_logger`, the training process's console output (logging, `print`s, warnings and the
  traceback of a crash) is uploaded to the MLflow run as `logs/console.log`, refreshed every 60 s while training
  runs (`console_upload_interval_seconds` to change it), so the log can be followed in the MLflow UI.
  The same fiddler logs a one-line progress summary (step, epoch/batch, `train_loss_step`, steps/s, samples/s,
  elapsed and remaining time) every 60 s (`progress_log_interval_seconds`), so progress shows in that log, and
  switches the progress bar to TQDM, because Lightning's default Rich bar is hidden once the console is captured.
  [#32](https://github.com/mlcast-community/mlcast/pull/32) (draft)
  Commits: [25b35f2](https://github.com/dmidk/mlcast/commit/25b35f269834eb1e7a2d884fe9d36a350e6fa659)
  [737d24e](https://github.com/dmidk/mlcast/commit/737d24e635239247e12d654a0a62d8a16a6dee0a) @leifdenby
- (DMI only) `dmi_500m_10min_convgru_experiment` config (`src/mlcast/config/dmi.py`): ConvGRU training on DMI's
  500 m / 10 min radar composite with its precomputed sampling index, 6 input / 18 forecast steps (1 h in → 3 h
  lead), fixed calendar-year splits (2016–2023 train / 2024 val / 2025 test), and logging to DMI's MLflow server.
  [416eaa9](https://github.com/dmidk/mlcast/commit/416eaa97352fcd93500f6aef30fb9c57076dae96)
  [c431746](https://github.com/dmidk/mlcast/commit/c4317463b07219b859b1d004f199b1ec5cc4ff5f) @leifdenby
- (DMI only) The DMI config saves a checkpoint every 3 h of training (Lightning's `ModelCheckpoint`
  `train_time_interval`), kept alongside the best-loss checkpoints as `every-3h-epoch=…-step=….ckpt`.
  [dmidk#3](https://github.com/dmidk/mlcast/pull/3) @leifdenby
- (DMI only) The DMI config's training split starts 2019-07-13 (was 2016-01-01) because of missing earlier data,
  and its sampling index moved to `/dmidata/projects/cloudphysics/mlcast/`.
  [214dbd5](https://github.com/dmidk/mlcast/commit/214dbd5b8ee075be2f21d26ffb61dc8cb30ee286) @arjj8
  [90b7899](https://github.com/dmidk/mlcast/commit/90b7899faa8276aaea89edbee6926f162f44caf3) @leifdenby
- Resuming training. `Experiment` has a `ckpt_path` passed to `Trainer.fit`, set by the new `resume_training`
  fiddler, which restores the full training state (so `max_time` is the total budget). `mlcast train --config
  mlflow:<run id, runs:/<id> or run URL>` continues an MLflow run: it downloads the run's config and latest
  checkpoint and logs the continuation as a new run named `<original>-from-step<N>`, linked to the original
  (`mlflow.parentRunId`, so the UI nests it, plus `mlcast.resumed_from_run` and
  `mlcast.resumed_from_checkpoint` tags), so the new run can log its own params and status; step numbers carry on.
  `--config mlflow-inplace:<run>` keeps logging into the original run instead. `use_mlflow_logger` now uploads
  every checkpoint to the run
  (`log_model="all"`); for older runs the local `<experiment_id>/<run_id>/checkpoints` is used. The config is
  now uploaded as `config/config.yaml`, as the log message already said, instead of a random temp-file name.
  @leifdenby

### Changed

- **Breaking:** `SourceDataPrecomputedSamplingDataset` is renamed `SourceDataIndexedDataset` and its
  `csv_path` argument is renamed `index_path`. It now reads a parquet sampling index; legacy `.csv` indexes are
  still accepted and used as-is.
  [#17](https://github.com/mlcast-community/mlcast/pull/17) @franchg @leifdenby @noralinn
- **Breaking:** the default experiment now loads `rainfall_flux` (was `rainfall_rate`) from
  `./data/sampled_datacubes.parquet`.
  [#17](https://github.com/mlcast-community/mlcast/pull/17) @franchg @leifdenby @noralinn
- **Breaking:** `target_mask` is now collapsed over the whole sequence to shape `(1, C, H, W)`. A cell is valid
  only if it is finite at every input and target step, and the masked loss broadcasts the mask over the forecast
  steps. [#17](https://github.com/mlcast-community/mlcast/pull/17) @franchg @leifdenby @noralinn
- (pending upstream) `mlcast build-sampling-index` now writes its output next to the Zarr dataset when `-o` is omitted (was the
  current working directory); pass `-o ./name.parquet` or an absolute path to write elsewhere.
  [#31](https://github.com/mlcast-community/mlcast/pull/31) (open)
  Commits: [219c84d](https://github.com/dmidk/mlcast/commit/219c84d14621c50b2fb8e7ba4bcace38dc5b4bd5) [14d2c34](https://github.com/dmidk/mlcast/commit/14d2c344724c7b84788d4b3917f716084ae09a8d) [9fb21a6](https://github.com/dmidk/mlcast/commit/9fb21a6ebcafd97f49624d3767b0128875fddeff) @leifdenby
- The CLI and `mlcast.data` import lazily, so `mlcast -h` and the data-prep commands start in well under a
  second. [#17](https://github.com/mlcast-community/mlcast/pull/17) @franchg @leifdenby @noralinn
- **Breaking:** `torch`, `torchvision` and `pytorch-lightning` are no longer core dependencies and are installed
  through the new `cpu` and `gpu-cu126` extras or the existing `gpu-cu128` / `gpu-cu130` extras
  (`uv sync --extra cpu`, `pip install "mlcast[cpu]"`). After a one-time `uv sync --extra ...`, plain `uv run`
  no longer swaps the installed PyTorch build. [#30](https://github.com/mlcast-community/mlcast/pull/30) @leifdenby
- (DMI only) `batch_size` raised from 16 to 32 in the DMI config; at 16 the A40 on `ohm.dmi.dk` used only ~43% of
  its GPU memory. [c14e44d](https://github.com/dmidk/mlcast/commit/c14e44d504a69390f52d04d4a4d910ef2c78473c) @leifdenby

### Fixed

- Checkpoint upload to MLflow could crash training: mlflow.dmidev.org cuts the connection on ~770 MB uploads,
  and Lightning's `MLFlowLogger` lets the error stop the run (a continuation died at its first validation). Upload
  is now opt-in (`use_mlflow_logger(upload_checkpoints=True)`, default off), and the new `ResilientMLFlowLogger`
  (used by `use_mlflow_logger` and `resume_training`, also for configs saved with plain `MLFlowLogger`) turns a
  failed upload into a warning and doesn't retry that checkpoint. @leifdenby
- A saved `config.yaml` with `(start, end)` date-range splits (e.g. the DMI config) can be built again: YAML has no
  tuples, so the ranges came back as lists and `validate_splits` rejected them, which broke `--config config.yaml`
  and `--config mlflow:<run>`. `SourceDataDataModule` now turns two-element lists back into tuples
  (`normalize_splits`). @leifdenby
- `mlcast train --config mlflow:<run>` failed with a 404 against tracking servers older than MLflow 3 (e.g.
  mlflow.dmidev.org): `MlflowClient.list_artifacts` in the MLflow 3 client also queries the logged-models endpoint.
  Artifacts are now listed and downloaded through the run's artifact repository instead. @leifdenby
- (pending upstream) A failed image upload in `NowcastLightningModule` (e.g. the MLflow proxy rejecting a large PNG with a 413) is
  now logged as a warning instead of killing the training run.
  [#23](https://github.com/mlcast-community/mlcast/pull/23) (open)
  Commits: [e543da2](https://github.com/dmidk/mlcast/commit/e543da2bc2c521af8ceccebd6c6c5821e21edbf7) @leifdenby
- (pending upstream) `ReflectivityScaling.normalize_reflectivity` now clips input to `[dbz_floor, dbz_ceiling]` before scaling, as
  documented, so values outside that range (e.g. a no-echo sentinel below `dbz_floor`) map into
  `[norm_min, norm_max]` instead of beyond it. [#25](https://github.com/mlcast-community/mlcast/pull/25) (open)
  Commits: [6570e05](https://github.com/dmidk/mlcast/commit/6570e053642734207540017abe2523b8a95de99c) @leifdenby
- (pending upstream) `train_from_config` no longer lets a single bad MLflow hyperparameter kill a training run. `log_hyperparams`
  truncates values but not keys and has no error handling, so an overlong key, an immutable-param rewrite, or a
  transient server error raised straight out of training before it started; the MLflow path is now routed
  through `mlcast.config.mlflow.log_hyperparams`, which truncates keys too and falls back to per-key logging
  (skipping and warning on any key that still fails) instead of raising. `use_mlflow_logger` also now accepts an
  explicit `tracking_uri`, so a caller that already knows the tracking server doesn't have to set
  `cfg.trainer.logger.tracking_uri` after the fact just to avoid a misleading "`MLFLOW_TRACKING_URI` is not set"
  warning. [#32](https://github.com/mlcast-community/mlcast/pull/32) (draft)
  Commits: [3a06293](https://github.com/dmidk/mlcast/commit/3a062936bd8631e061d1eb0d37fc8abc3b37c5a8) [cb5d411](https://github.com/dmidk/mlcast/commit/cb5d41102473bbccc2f25a64a5c8bd370259abda) @leifdenby
- Time-split subsets of an indexed dataset now rebase the absolute `t` indexes onto the sliced store and drop
  windows that would cross the split boundary, so they no longer read the wrong time steps or leak samples
  across splits. [#17](https://github.com/mlcast-community/mlcast/pull/17) @franchg @leifdenby @noralinn
- (pending upstream) `mlcast build-sampling-index` now opens its output before scanning and creates missing parent directories,
  so an unwritable output (e.g. disk quota exceeded) fails immediately instead of after the scan starts. A write
  failure mid-scan now aborts the run rather than leaving it hanging, and the output path is logged as an
  absolute path. [#31](https://github.com/mlcast-community/mlcast/pull/31) (open)
  Commits: [f79ac21](https://github.com/dmidk/mlcast/commit/f79ac21494e4d4c5809268a6112100c37c6ee70e) @leifdenby

### Maintenance

- `bottleneck`, `pyarrow` and `pydantic` are now core dependencies; the unused `fire` dependency is removed.
  [#17](https://github.com/mlcast-community/mlcast/pull/17) @franchg @leifdenby @noralinn
- CI check that fails pull requests which don't update `CHANGELOG.md`; PRs that don't need an entry can opt out
  with the `no changelog` label. [#26](https://github.com/mlcast-community/mlcast/pull/26) @leifdenby
- CI workflow that runs the pytest suite on every push and pull request.
  [#30](https://github.com/mlcast-community/mlcast/pull/30) @leifdenby
- Missing changelog entries for [#17](https://github.com/mlcast-community/mlcast/pull/17).
  [#26](https://github.com/mlcast-community/mlcast/pull/26) @leifdenby
- Register `equivalent_reflectivity_factor` (radar reflectivity in dBZ) in the normalization registries, so
  training on it no longer fails with a `KeyError`, and drop the non-CF `radar_reflectivity` name from the
  sampler. [#24](https://github.com/mlcast-community/mlcast/pull/24) @leifdenby

## v0.1.0 - 2026-06-01

This release establishes `mlcast` as a usable foundation for machine-learning weather nowcasting experiments.
It brings together configurable experiment construction, Zarr-backed meteorological data loading, a PyTorch
Lightning training loop, probabilistic ConvGRU nowcasting, and reproducible run configuration. With this
release, users can train the built-in ConvGRU baseline from the command line, customize experiments through
Fiddle configs and fiddlers, or replace the network with their own nowcasting architecture while keeping the
surrounding data, training, logging, and reproducibility machinery.

### Added

- Fiddle-based experiment configuration centered on the default `training_experiment` graph, covering dataset
  setup, data module, network, Lightning module, optimizer, scheduler, callbacks, logger, and trainer.
- `mlcast train` CLI for launching the default experiment, applying ordered `set:` overrides and `fiddler:`
  mutations, loading saved YAML configs, switching config functions, and inspecting resolved configs before
  training.
- Python API for constructing, editing, validating, building, and running experiment configs programmatically.
- Zarr-backed source-data pipeline with CF-standard-name variable lookup, normalization, precomputed CSV
  sampling, random on-the-fly sampling, train/validation/test splitting, spatial augmentations, optional target
  masks, and anonymous S3 access.
- Precipitation conversion and normalization utilities for rainfall rate, rainfall flux, and 5-minute rainfall
  amount.
- Generic PyTorch Lightning nowcasting module that wraps any compatible network and handles training,
  validation, testing, optimizer/scheduler construction, ensemble forecasting, masked losses, image logging,
  checkpoint loading, and prediction.
- Built-in ConvGRU encoder-decoder nowcasting model with multi-scale spatial encoding/decoding, latent-space
  forecast rollout, padding for arbitrary spatial sizes, and optional stochastic ensemble generation.
- Loss functions for deterministic and probabilistic nowcasting, including MSE, MAE, CRPS, almost-fair CRPS,
  temporal consistency regularization, and masking of invalid target pixels.
- Semantic fiddlers for common multi-parameter configuration changes, including random sampling, variable
  selection, masking, ratio splits, anonymous S3 datasets, and MLflow logging.
- Reproducibility and experiment tracking support through saved or uploaded Fiddle YAML configs, flattened config
  hyperparameters, TensorBoard logging, MLflow logging, W&B-compatible config artifact handling, and MLflow
  system metadata/metrics logging.
- Documentation covering installation, CLI workflows, Python workflows, custom network integration, available
  fiddlers, project structure, and the ConvGRU architecture.

### Changed

- Training now centers on a single Fiddle-based experiment graph rather than the earlier ad hoc configuration
  flow.
- The CLI now uses `mlcast train` with nested `--config` overrides and named fiddlers for more reproducible
  experiment changes.
- The default training setup now uses the ConvGRU ensemble nowcasting pipeline as the primary supported
  baseline.
- Saved experiment configs are now persisted alongside runs for exact reproduction.

## 0.0.1a4 - 2025-08-28

This was the first published PyPI release of `mlcast`, establishing the initial installable package baseline with
the early nowcasting training pipeline, core package layout, and initial public documentation.
