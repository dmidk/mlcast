"""Fiddle-based experiment configuration package.

This package defines the configuration schemas, validation constraints,
and runtime orchestration logic for `mlcast`.
"""

from .base import Experiment, training_experiment
from .consistency_checks import validate_config
from .dmi import dmi_500m_10min_convgru_experiment
from .fiddlers import (
    resume_training,
    set_variables,
    toggle_masking,
    use_anon_s3_dataset,
    use_mlflow_logger,
    use_random_sampler,
    use_ratio_splits,
)
from .loader import load_yaml_config
from .orchestrator import train_from_config

__all__ = [
    "Experiment",
    "training_experiment",
    "dmi_500m_10min_convgru_experiment",
    "validate_config",
    "train_from_config",
    "load_yaml_config",
    "set_variables",
    "toggle_masking",
    "use_random_sampler",
    "use_ratio_splits",
    "use_mlflow_logger",
    "use_anon_s3_dataset",
    "resume_training",
]
