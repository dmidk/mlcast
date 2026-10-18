"""Tests for resuming training from a checkpoint or an MLflow run."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import fiddle as fdl
import pytest
from pytorch_lightning.loggers import MLFlowLogger

from mlcast.__main__ import _extract_mlflow_run_ref
from mlcast.config import Experiment, resume_training, training_experiment, use_mlflow_logger
from mlcast.config.mlflow_resume import parse_run_ref, resolve_resume_source

RUN_ID = "52dd4f17fad94d2eb0ac8489c234f6c2"


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        (RUN_ID, (None, RUN_ID)),
        (f"runs:/{RUN_ID}", (None, RUN_ID)),
        (f"runs:/{RUN_ID}/model", (None, RUN_ID)),
        (f"https://mlflow.example.org/#/experiments/114/runs/{RUN_ID}", ("https://mlflow.example.org", RUN_ID)),
        (
            f"https://mlflow.example.org/#/experiments/114/runs/{RUN_ID}/model-metrics",
            ("https://mlflow.example.org", RUN_ID),
        ),
        (f"http://localhost:5000/#/experiments/1/runs/{RUN_ID}", ("http://localhost:5000", RUN_ID)),
    ],
)
def test_parse_run_ref(ref: str, expected: tuple[str | None, str]) -> None:
    assert parse_run_ref(ref) == expected


@pytest.mark.parametrize("ref", ["not-a-run", "runs:/abc", "https://mlflow.example.org/#/experiments/114"])
def test_parse_run_ref_rejects_garbage(ref: str) -> None:
    with pytest.raises(ValueError, match="Not an MLflow run reference"):
        parse_run_ref(ref)


def _fake_client(artifacts: dict[str, list[tuple[str, bool]]], experiment_id: str = "114") -> MagicMock:
    """MlflowClient stand-in: `artifacts` maps a directory ("" for the root) to (path, is_dir) entries."""
    client = MagicMock()
    client.get_run.return_value = SimpleNamespace(info=SimpleNamespace(experiment_id=experiment_id))
    client.list_artifacts.side_effect = lambda run_id, path=None: [
        SimpleNamespace(path=p, is_dir=d) for p, d in artifacts.get(path or "", [])
    ]

    def download(run_id: str, path: str, dst: str) -> str:
        out = Path(dst) / path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(f"downloaded {path}")
        return str(out)

    client.download_artifacts.side_effect = download
    return client


def test_resolve_picks_latest_logged_checkpoint(tmp_path: Path) -> None:
    client = _fake_client(
        {
            "": [
                ("config", True),
                ("logs", True),
                ("train", True),
                ("epoch=0-step=4999", True),
                ("every-3h-epoch=0-step=11034", True),
                ("every-3h-epoch=0-step=79512", True),
            ],
            "config": [("config/config.yaml", False)],
            "epoch=0-step=4999": [
                ("epoch=0-step=4999/epoch=0-step=4999.ckpt", False),
                ("epoch=0-step=4999/metadata.yaml", False),
            ],
            "every-3h-epoch=0-step=11034": [("every-3h-epoch=0-step=11034/every-3h-epoch=0-step=11034.ckpt", False)],
            "every-3h-epoch=0-step=79512": [("every-3h-epoch=0-step=79512/every-3h-epoch=0-step=79512.ckpt", False)],
        }
    )

    source = resolve_resume_source(RUN_ID, download_dir=tmp_path, local_root=tmp_path / "nowhere", client=client)

    assert source.run_id == RUN_ID
    assert source.config_path == tmp_path / RUN_ID / "config" / "config.yaml"
    assert source.ckpt_path.name == "every-3h-epoch=0-step=79512.ckpt"
    downloaded = [c.args[1] for c in client.download_artifacts.call_args_list]
    assert downloaded == ["config/config.yaml", "every-3h-epoch=0-step=79512/every-3h-epoch=0-step=79512.ckpt"]


def test_resolve_reuses_already_downloaded_checkpoint(tmp_path: Path) -> None:
    ckpt = "epoch=0-step=10/epoch=0-step=10.ckpt"
    client = _fake_client(
        {
            "": [("config", True), ("epoch=0-step=10", True)],
            "config": [("config/config.yaml", False)],
            "epoch=0-step=10": [(ckpt, False)],
        }
    )
    existing = tmp_path / RUN_ID / ckpt
    existing.parent.mkdir(parents=True)
    existing.write_text("already here")

    source = resolve_resume_source(RUN_ID, download_dir=tmp_path, client=client)

    assert source.ckpt_path == existing
    assert [c.args[1] for c in client.download_artifacts.call_args_list] == ["config/config.yaml"]


def test_resolve_falls_back_to_local_checkpoints(tmp_path: Path) -> None:
    client = _fake_client({"": [("config", True), ("logs", True)], "config": [("config/tmpabc.yaml", False)]})
    ckpt_dir = tmp_path / "root" / "114" / RUN_ID / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    for name in ["epoch=0-step=4999.ckpt", "every-3h-epoch=0-step=79512.ckpt", "every-3h-epoch=0-step=11034.ckpt"]:
        (ckpt_dir / name).write_text("ckpt")

    source = resolve_resume_source(RUN_ID, download_dir=tmp_path / "dl", local_root=tmp_path / "root", client=client)

    assert source.ckpt_path == ckpt_dir / "every-3h-epoch=0-step=79512.ckpt"


def test_resolve_without_any_checkpoint_raises(tmp_path: Path) -> None:
    client = _fake_client({"": [("config", True)], "config": [("config/config.yaml", False)]})
    with pytest.raises(FileNotFoundError, match="no checkpoints"):
        resolve_resume_source(RUN_ID, download_dir=tmp_path, local_root=tmp_path, client=client)


def test_resume_training_fiddler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.org/")
    cfg = training_experiment.as_buildable()
    use_mlflow_logger(cfg)

    resume_training(cfg, ckpt_path="/ckpts/last.ckpt", mlflow_run_id=RUN_ID)

    assert cfg.ckpt_path == "/ckpts/last.ckpt"
    assert cfg.trainer.logger.run_id == RUN_ID
    assert cfg.trainer.logger.log_model == "all"


def test_resume_training_fiddler_without_mlflow_logger() -> None:
    cfg = training_experiment.as_buildable()

    resume_training(cfg, ckpt_path="/ckpts/last.ckpt")
    assert cfg.ckpt_path == "/ckpts/last.ckpt"

    with pytest.raises(ValueError, match="needs an MLFlowLogger"):
        resume_training(cfg, ckpt_path="/ckpts/last.ckpt", mlflow_run_id=RUN_ID)


def test_use_mlflow_logger_uploads_checkpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.org/")
    cfg = training_experiment.as_buildable()

    use_mlflow_logger(cfg)

    assert cfg.trainer.logger.__fn_or_cls__ is MLFlowLogger
    assert cfg.trainer.logger.log_model == "all"


@pytest.mark.parametrize("ckpt_path", [None, "/ckpts/last.ckpt"])
def test_experiment_run_passes_ckpt_path(ckpt_path: str | None) -> None:
    trainer = MagicMock()
    experiment = Experiment(pl_module=MagicMock(), data=MagicMock(), trainer=trainer, ckpt_path=ckpt_path)

    experiment.run()

    assert trainer.fit.call_args.kwargs["ckpt_path"] == ckpt_path


def test_experiment_ckpt_path_settable_in_config() -> None:
    cfg = training_experiment.as_buildable()
    assert fdl.Config(Experiment, pl_module=None, data=None, trainer=None).ckpt_path is None
    cfg.ckpt_path = "/ckpts/last.ckpt"
    assert cfg.ckpt_path == "/ckpts/last.ckpt"


@pytest.mark.parametrize(
    "argv",
    [
        ["--config", f"mlflow:{RUN_ID}", "--config", "set:trainer.max_time='01:00:00:00'"],
        [f"--config=mlflow:{RUN_ID}", "--config", "set:trainer.max_time='01:00:00:00'"],
    ],
)
def test_extract_mlflow_run_ref(argv: list[str]) -> None:
    run_ref, remaining = _extract_mlflow_run_ref(argv)

    assert run_ref == RUN_ID
    assert remaining == ["--config", "set:trainer.max_time='01:00:00:00'"]


def test_extract_mlflow_run_ref_absent() -> None:
    argv = ["--config", "config:dmi_500m_10min_convgru_experiment"]
    assert _extract_mlflow_run_ref(argv) == (None, argv)
