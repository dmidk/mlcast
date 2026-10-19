"""Resolve an MLflow run into a config file and a checkpoint to resume training from.

Used by ``mlcast train --config mlflow:<run>``. A run can be referred to by
its id, as ``runs:/<id>``, or by its URL in the MLflow UI (which also gives
the tracking server). The run's config YAML (logged under ``config/`` by
`train_from_config`) and its latest checkpoint are downloaded locally.

Checkpoints are found in two places, in this order:

1. the run's artifacts, where ``MLFlowLogger(log_model="all")``
   (``use_mlflow_logger(upload_checkpoints=True)``) uploads each checkpoint
   into a directory named after it (``<stem>/<stem>.ckpt``);
2. the run's local checkpoint directory,
   ``<default_root_dir>/<experiment_id>/<run_id>/checkpoints``, for runs that
   didn't upload checkpoints (only on the machine that trained).

"Latest" is the checkpoint with the highest ``step=N`` in its filename,
which every Lightning ``ModelCheckpoint`` filename in mlcast includes.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from loguru import logger
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.tracking import MlflowClient

# Directories that `train_from_config` and the callbacks log to; never checkpoints.
_NON_CHECKPOINT_DIRS = {"config", "logs", "train"}
_RUN_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_STEP_RE = re.compile(r"step=(\d+)")


@dataclass
class ResumeSource:
    """Everything needed to resume a run."""

    run_id: str
    tracking_uri: str | None
    config_path: Path
    ckpt_path: Path
    run_name: str | None = None

    @property
    def step(self) -> int:
        """Training step of the checkpoint, from its ``step=N`` filename part (-1 if absent)."""
        return _step(self.ckpt_path.name)


def parse_run_ref(ref: str) -> tuple[str | None, str]:
    """Split a run reference into ``(tracking_uri, run_id)``.

    Parameters
    ----------
    ref : str
        A run id, ``runs:/<id>[/...]``, or an MLflow UI URL such as
        ``https://mlflow.example.org/#/experiments/114/runs/<id>/metrics``.

    Returns
    -------
    tracking_uri : str or None
        The server from a URL, else ``None`` (use ``MLFLOW_TRACKING_URI``).
    run_id : str
        The 32-character run id.
    """
    if _RUN_ID_RE.match(ref):
        return None, ref
    if ref.startswith("runs:/"):
        run_id = ref[len("runs:/") :].split("/")[0]
        if _RUN_ID_RE.match(run_id):
            return None, run_id
    parsed = urlparse(ref)
    if parsed.scheme in ("http", "https"):
        # The UI routes live in the fragment: #/experiments/<e>/runs/<id>/...
        match = re.search(r"/runs/([0-9a-f]{32})", parsed.fragment or parsed.path)
        if match:
            base_path = parsed.path.rstrip("/")
            return f"{parsed.scheme}://{parsed.netloc}{base_path}", match.group(1)
    raise ValueError(f"Not an MLflow run reference: {ref!r} (expected a run id, runs:/<id> or a run URL)")


def _step(name: str) -> int:
    match = _STEP_RE.search(name)
    return int(match.group(1)) if match else -1


def _remote_checkpoints(repo: ArtifactRepository) -> list[str]:
    """Return artifact paths of the ``.ckpt`` files logged to the run."""
    paths = []
    for entry in repo.list_artifacts():
        if not entry.is_dir or entry.path in _NON_CHECKPOINT_DIRS:
            continue
        paths += [f.path for f in repo.list_artifacts(entry.path) if f.path.endswith(".ckpt")]
    return paths


def _local_checkpoints(root: Path, experiment_id: str, run_id: str) -> list[Path]:
    return sorted((root / experiment_id / run_id / "checkpoints").glob("*.ckpt"))


def resolve_resume_source(
    ref: str,
    download_dir: Path,
    local_root: Path | None = None,
    client: MlflowClient | None = None,
    repo: ArtifactRepository | None = None,
) -> ResumeSource:
    """Download the config and pick the latest checkpoint of an MLflow run.

    Parameters
    ----------
    ref : str
        Run reference, see `parse_run_ref`.
    download_dir : Path
        Where to download artifacts; a ``<run_id>`` subdirectory is used.
        Files already there are reused.
    local_root : Path, optional
        Root of local checkpoint directories (the trainer's
        ``default_root_dir``); defaults to the current directory.
    client : MlflowClient, optional
        Client to use (for tests); built from the reference otherwise.
    repo : ArtifactRepository, optional
        The run's artifact repository (for tests); built from the run's
        ``artifact_uri`` otherwise.

    Returns
    -------
    ResumeSource
        Run id, tracking URI, and local paths of the config and checkpoint.
    """
    tracking_uri, run_id = parse_run_ref(ref)
    effective_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if client is None:
        client = MlflowClient(tracking_uri=effective_uri)
    run = client.get_run(run_id)
    if repo is None:
        # List and download through the run's artifact repository, not
        # `MlflowClient.list_artifacts(run_id, ...)`: in MLflow 3 the latter
        # also queries the logged-models endpoint, which older tracking
        # servers (e.g. mlflow.dmidev.org) don't have, so it fails with a 404.
        repo = get_artifact_repository(run.info.artifact_uri, tracking_uri=effective_uri)
    dest = download_dir / run_id
    dest.mkdir(parents=True, exist_ok=True)

    configs = [f.path for f in repo.list_artifacts("config") if f.path.endswith((".yaml", ".yml"))]
    if not configs:
        raise FileNotFoundError(f"MLflow run {run_id} has no config YAML under config/")
    # Runs from before the config was uploaded under a fixed name have a
    # temp-named YAML; after an in-place resume they also have config.yaml,
    # which is the newer one.
    config_artifact = "config/config.yaml" if "config/config.yaml" in configs else configs[0]
    if len(configs) > 1:
        logger.warning(f"MLflow run {run_id} has several config files {configs}; using {config_artifact}")
    config_path = Path(repo.download_artifacts(config_artifact, str(dest)))

    remote = _remote_checkpoints(repo)
    if remote:
        latest = max(remote, key=_step)
        local_copy = dest / latest
        if local_copy.exists():
            logger.info(f"Reusing downloaded checkpoint {local_copy}")
            ckpt_path = local_copy
        else:
            logger.info(f"Downloading checkpoint {latest} from MLflow run {run_id}")
            ckpt_path = Path(repo.download_artifacts(latest, str(dest)))
    else:
        local = _local_checkpoints(local_root or Path.cwd(), run.info.experiment_id, run_id)
        if not local:
            raise FileNotFoundError(
                f"MLflow run {run_id} has no checkpoints logged as artifacts, and none were found locally in "
                f"{(local_root or Path.cwd()) / run.info.experiment_id / run_id / 'checkpoints'}"
            )
        ckpt_path = max(local, key=lambda p: _step(p.name))
        logger.info(f"No checkpoints logged to MLflow run {run_id}; using local checkpoint {ckpt_path}")

    logger.info(f"Resuming MLflow run {run_id} from {ckpt_path.name} (step {_step(ckpt_path.name)})")
    return ResumeSource(
        run_id=run_id,
        tracking_uri=tracking_uri,
        config_path=config_path,
        ckpt_path=ckpt_path,
        run_name=getattr(run.info, "run_name", None),
    )
