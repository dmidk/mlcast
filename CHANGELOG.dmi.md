# DMI changelog

Changes specific to the DMI training branch (`dev/dmi-convgru-training` on
[dmidk/mlcast](https://github.com/dmidk/mlcast)).

Changes that are also going upstream to
[mlcast-community/mlcast](https://github.com/mlcast-community/mlcast) are recorded in
[`CHANGELOG.md`](CHANGELOG.md) through their upstream PRs. This file only covers what stays at DMI, plus which
upstream work has been brought in early. Keeping DMI entries out of `CHANGELOG.md` avoids merge conflicts every
time upstream work is merged in.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Added

- `dmi_500m_10min_convgru_experiment` config (`src/mlcast/config/dmi.py`): ConvGRU training on DMI's 500 m /
  10 min radar composite with its precomputed sampling index, 6 input / 18 forecast steps (1 h in → 3 h lead),
  fixed calendar-year splits (2016–2023 train / 2024 val / 2025 test), and logging to DMI's MLflow server.
  @leifdenby
- The MLflow console-log upload from upstream
  [mlcast-community/mlcast#32](https://github.com/mlcast-community/mlcast/pull/32), brought in ahead of upstream.
  With `fiddler:use_mlflow_logger`, training runs upload their console output as `logs/console.log`.
  [dmidk/mlcast#1](https://github.com/dmidk/mlcast/pull/1) @leifdenby

### Changed

- `batch_size` raised from 16 to 32 in the DMI config: at 16, the A40 on `ohm.dmi.dk` used only ~43% of its GPU
  memory. @leifdenby
