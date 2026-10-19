"""Tests for resuming training from a checkpoint or an MLflow run."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import fiddle as fdl
import pytest
from loguru import logger as log
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from mlcast.__main__ import _extract_mlflow_run_ref
from mlcast.config import Experiment, mlflow_resume, resume_training, training_experiment, use_mlflow_logger
from mlcast.config.mlflow_resume import parse_run_ref, resolve_resume_source
from mlcast.loggers import ResilientMLFlowLogger

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


ARTIFACT_URI = f"mlflow-artifacts:/114/{RUN_ID}/artifacts"


def _fake_client(experiment_id: str = "114") -> MagicMock:
    """MlflowClient stand-in that only knows about the run itself."""
    client = MagicMock()
    client.get_run.return_value = SimpleNamespace(
        info=SimpleNamespace(experiment_id=experiment_id, artifact_uri=ARTIFACT_URI)
    )
    return client


def _fake_repo(artifacts: dict[str, list[tuple[str, bool]]]) -> MagicMock:
    """ArtifactRepository stand-in: `artifacts` maps a directory ("" for the root) to (path, is_dir) entries."""
    repo = MagicMock()
    repo.list_artifacts.side_effect = lambda path=None: [
        SimpleNamespace(path=p, is_dir=d) for p, d in artifacts.get(path or "", [])
    ]

    def download(path: str, dst: str) -> str:
        out = Path(dst) / path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(f"downloaded {path}")
        return str(out)

    repo.download_artifacts.side_effect = download
    return repo


def test_resolve_picks_latest_logged_checkpoint(tmp_path: Path) -> None:
    repo = _fake_repo(
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

    source = resolve_resume_source(
        RUN_ID, download_dir=tmp_path, local_root=tmp_path / "nowhere", client=_fake_client(), repo=repo
    )

    assert source.run_id == RUN_ID
    assert source.config_path == tmp_path / RUN_ID / "config" / "config.yaml"
    assert source.ckpt_path.name == "every-3h-epoch=0-step=79512.ckpt"
    downloaded = [c.args[0] for c in repo.download_artifacts.call_args_list]
    assert downloaded == ["config/config.yaml", "every-3h-epoch=0-step=79512/every-3h-epoch=0-step=79512.ckpt"]


def test_resolve_lists_artifacts_via_the_runs_artifact_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Artifacts go through the repository built from the run's artifact_uri, never
    `MlflowClient.list_artifacts`, which in MLflow 3 also calls a logged-models
    endpoint that older tracking servers lack (404)."""
    client = _fake_client()
    client.list_artifacts.side_effect = AssertionError("MlflowClient.list_artifacts must not be used")
    repo = _fake_repo({"": [("config", True)], "config": [("config/config.yaml", False)]})
    built_with = {}

    def fake_get_artifact_repository(artifact_uri: str, tracking_uri: str | None = None) -> MagicMock:
        built_with.update(artifact_uri=artifact_uri, tracking_uri=tracking_uri)
        return repo

    monkeypatch.setattr(mlflow_resume, "get_artifact_repository", fake_get_artifact_repository)
    ckpt_dir = tmp_path / "114" / RUN_ID / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "epoch=0-step=10.ckpt").write_text("ckpt")

    resolve_resume_source(
        f"https://mlflow.example.org/#/experiments/114/runs/{RUN_ID}",
        download_dir=tmp_path / "dl",
        local_root=tmp_path,
        client=client,
    )

    assert built_with == {"artifact_uri": ARTIFACT_URI, "tracking_uri": "https://mlflow.example.org"}


def test_resolve_reuses_already_downloaded_checkpoint(tmp_path: Path) -> None:
    ckpt = "epoch=0-step=10/epoch=0-step=10.ckpt"
    repo = _fake_repo(
        {
            "": [("config", True), ("epoch=0-step=10", True)],
            "config": [("config/config.yaml", False)],
            "epoch=0-step=10": [(ckpt, False)],
        }
    )
    existing = tmp_path / RUN_ID / ckpt
    existing.parent.mkdir(parents=True)
    existing.write_text("already here")

    source = resolve_resume_source(RUN_ID, download_dir=tmp_path, client=_fake_client(), repo=repo)

    assert source.ckpt_path == existing
    assert [c.args[0] for c in repo.download_artifacts.call_args_list] == ["config/config.yaml"]


def test_resolve_falls_back_to_local_checkpoints(tmp_path: Path) -> None:
    repo = _fake_repo({"": [("config", True), ("logs", True)], "config": [("config/tmpabc.yaml", False)]})
    ckpt_dir = tmp_path / "root" / "114" / RUN_ID / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    for name in ["epoch=0-step=4999.ckpt", "every-3h-epoch=0-step=79512.ckpt", "every-3h-epoch=0-step=11034.ckpt"]:
        (ckpt_dir / name).write_text("ckpt")

    source = resolve_resume_source(
        RUN_ID, download_dir=tmp_path / "dl", local_root=tmp_path / "root", client=_fake_client(), repo=repo
    )

    assert source.ckpt_path == ckpt_dir / "every-3h-epoch=0-step=79512.ckpt"


def test_resolve_without_any_checkpoint_raises(tmp_path: Path) -> None:
    repo = _fake_repo({"": [("config", True)], "config": [("config/config.yaml", False)]})
    with pytest.raises(FileNotFoundError, match="no checkpoints"):
        resolve_resume_source(RUN_ID, download_dir=tmp_path, local_root=tmp_path, client=_fake_client(), repo=repo)


def test_resume_training_fiddler_starts_a_linked_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.org/")
    cfg = training_experiment.as_buildable()
    use_mlflow_logger(cfg)

    resume_training(
        cfg, ckpt_path="/ckpts/epoch=0-step=83299.ckpt", mlflow_run_id=RUN_ID, run_name="sow-from-step83299"
    )

    assert cfg.ckpt_path == "/ckpts/epoch=0-step=83299.ckpt"
    assert cfg.trainer.logger.run_id is None
    assert cfg.trainer.logger.run_name == "sow-from-step83299"
    assert cfg.trainer.logger.tags == {
        "mlflow.parentRunId": RUN_ID,
        "mlcast.resumed_from_run": RUN_ID,
        "mlcast.resumed_from_checkpoint": "epoch=0-step=83299.ckpt",
    }
    assert cfg.trainer.logger.log_model is False
    assert cfg.trainer.logger.__fn_or_cls__ is ResilientMLFlowLogger


def test_resume_training_fiddler_drops_ids_inherited_from_a_resumed_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """A config saved by an earlier resume carries that run's id, name and links."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.org/")
    cfg = training_experiment.as_buildable()
    use_mlflow_logger(cfg)
    cfg.trainer.logger.run_id = "a" * 32
    cfg.trainer.logger.run_name = "old-from-step10"
    cfg.trainer.logger.tags = {"mlflow.parentRunId": "b" * 32, "team": "dmi"}

    resume_training(cfg, ckpt_path="/ckpts/x-step=20.ckpt", mlflow_run_id=RUN_ID)

    assert cfg.trainer.logger.run_id is None
    assert cfg.trainer.logger.run_name is None
    assert cfg.trainer.logger.tags["mlflow.parentRunId"] == RUN_ID
    assert cfg.trainer.logger.tags["team"] == "dmi"


def test_resume_training_fiddler_inplace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.org/")
    cfg = training_experiment.as_buildable()
    use_mlflow_logger(cfg)

    resume_training(cfg, ckpt_path="/ckpts/last.ckpt", mlflow_run_id=RUN_ID, inplace=True, run_name="ignored")

    assert cfg.trainer.logger.run_id == RUN_ID
    assert cfg.trainer.logger.tags is None
    assert cfg.trainer.logger.run_name is None


def test_resume_training_fiddler_without_mlflow_logger() -> None:
    cfg = training_experiment.as_buildable()

    resume_training(cfg, ckpt_path="/ckpts/last.ckpt")
    assert cfg.ckpt_path == "/ckpts/last.ckpt"

    with pytest.raises(ValueError, match="needs an MLFlowLogger"):
        resume_training(cfg, ckpt_path="/ckpts/last.ckpt", mlflow_run_id=RUN_ID)


def test_resume_training_keeps_upload_setting_and_makes_old_configs_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    """A config saved by an older run (plain MLFlowLogger, log_model="all") resumes with the
    resilient logger, so a refused checkpoint upload can't crash it; the upload setting is kept."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.org/")
    cfg = training_experiment.as_buildable()
    cfg.trainer.logger = fdl.Config(MLFlowLogger, experiment_name="x", log_model="all")

    resume_training(cfg, ckpt_path="/ckpts/last.ckpt", mlflow_run_id=RUN_ID)

    assert cfg.trainer.logger.__fn_or_cls__ is ResilientMLFlowLogger
    assert cfg.trainer.logger.log_model == "all"
    assert cfg.trainer.logger.experiment_name == "x"


@pytest.mark.parametrize(("upload", "log_model"), [(False, False), (True, "all")])
def test_use_mlflow_logger_checkpoint_upload_is_opt_in(
    monkeypatch: pytest.MonkeyPatch, upload: bool, log_model: bool | str
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.org/")
    cfg = training_experiment.as_buildable()

    if upload:
        use_mlflow_logger(cfg, upload_checkpoints=True)
    else:
        use_mlflow_logger(cfg)

    assert cfg.trainer.logger.__fn_or_cls__ is ResilientMLFlowLogger
    assert cfg.trainer.logger.log_model == log_model


def test_resilient_logger_turns_upload_failure_into_warning(tmp_path: Path) -> None:
    """A refused upload is a warning, and the checkpoint isn't retried on the next save."""
    ckpt = tmp_path / "epoch=1-step=88299.ckpt"
    ckpt.write_text("ckpt")
    logger = ResilientMLFlowLogger(experiment_name="x", tracking_uri=f"file://{tmp_path}/mlruns", log_model="all")
    client = MagicMock()
    client.log_artifact.side_effect = ConnectionError("EOF occurred in violation of protocol")
    logger._mlflow_client = client
    logger._run_id = RUN_ID
    logger._initialized = True
    callback = MagicMock(spec=ModelCheckpoint)
    callback.last_model_path = ""
    callback.best_model_path = str(ckpt)
    callback.best_model_score = None
    callback.best_k_models = {}
    callback.save_top_k = 1
    messages: list[str] = []
    sink = log.add(lambda m: messages.append(m.record["message"]), level="WARNING")
    try:
        logger.after_save_checkpoint(callback)
        logger.after_save_checkpoint(callback)
    finally:
        log.remove(sink)

    assert client.log_artifact.call_count == 1
    assert len(messages) == 1
    assert "Could not upload" in messages[0] and "EOF occurred" in messages[0]


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
    run_ref, inplace, remaining = _extract_mlflow_run_ref(argv)

    assert run_ref == RUN_ID
    assert inplace is False
    assert remaining == ["--config", "set:trainer.max_time='01:00:00:00'"]


@pytest.mark.parametrize("argv", [["--config", f"mlflow-inplace:{RUN_ID}"], [f"--config=mlflow-inplace:{RUN_ID}"]])
def test_extract_mlflow_run_ref_inplace(argv: list[str]) -> None:
    assert _extract_mlflow_run_ref(argv) == (RUN_ID, True, [])


def test_extract_mlflow_run_ref_absent() -> None:
    argv = ["--config", "config:dmi_500m_10min_convgru_experiment"]
    assert _extract_mlflow_run_ref(argv) == (None, False, argv)


def test_resolve_prefers_config_yaml_and_reports_run_name_and_step(tmp_path: Path) -> None:
    client = _fake_client()
    client.get_run.return_value.info.run_name = "intrigued-sow-455"
    repo = _fake_repo(
        {
            "": [("config", True), ("epoch=0-step=83299", True)],
            "config": [("config/tmpm6d5s5g9.yaml", False), ("config/config.yaml", False)],
            "epoch=0-step=83299": [("epoch=0-step=83299/epoch=0-step=83299.ckpt", False)],
        }
    )

    source = resolve_resume_source(RUN_ID, download_dir=tmp_path, client=client, repo=repo)

    assert source.config_path.name == "config.yaml"
    assert source.run_name == "intrigued-sow-455"
    assert source.step == 83299
