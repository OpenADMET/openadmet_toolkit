import gc
import os
import tempfile
import pandas as pd
from pathlib import Path
from shutil import copyfile
from typing import Optional, Union
import subprocess

import numpy as np
import torch
from loguru import logger

from pydantic import Field, field_validator

from openadmet.toolkit.cofolding.cofold_base import CoFoldingEngine


try:
    from openfold3.core.config import config_utils
    from openfold3.core.data.pipelines.preprocessing.template import TemplatePreprocessor
    from openfold3.core.data.tools.colabfold_msa_server import preprocess_colabfold_msas
    from openfold3.entry_points.experiment_runner import (
        InferenceExperimentRunner,
        TrainingExperimentRunner,
    )
    from openfold3.entry_points.validator import (
        InferenceExperimentConfig,
        TrainingExperimentConfig,
    )
    from openfold3.projects.of3_all_atom.config.dataset_config_components import (
        colabfold_msa_settings,
    )
    from openfold3.projects.of3_all_atom.config.inference_query_format import (
        InferenceQuerySet,
    )
except ImportError:
    HAS_OPENFOLD3 = False
else:
    HAS_OPENFOLD3 = True


class OpenFold3CofoldingEngine(CoFoldingEngine):

    use_msa_server: bool = Field(
                False, description="Use MSA server for multiple sequence alignment"
    )
    num_diffusion_samples: int = Field(None, description="Number of diffusion samples to generate for each query")

    num_model_seeds: int = Field(None, description="Number of model seeds to use for each query")

    use_msa_server: bool = Field(True, description="Use MSA server for multiple sequence alignment")

    use_templates: bool = Field(False, description="Use templates for structure prediction")

    inference_ckpt_path: Path = Field(description="Path to the inference checkpoint")


    @field_validator("inference_ckpt_path")
    def check_inference_ckpt_path(cls, v):
        # path must exist
        if not v.exists():
            raise ValueError(f"inference_ckpt_path must exist, got {v}")
        return v

    def inference(
        self,
        query_json: Path,
        runner_yaml: Path | None = None,
    ):
        if not HAS_OPENFOLD3:
            raise ImportError("OpenFold3 is not installed.")

        runner_args = config_utils.load_yaml(runner_yaml) if runner_yaml else {}

        expt_config = InferenceExperimentConfig(
            inference_ckpt_path=self.inference_ckpt_path, **runner_args
        )
        expt_runner = InferenceExperimentRunner(
            expt_config,
            self.num_diffusion_samples,
            self.num_model_seeds,
            self.use_msa_server,
            self.use_templates,
            self.output_dir,
        )

        # Dump experiment runner
        import json

        with open(self.output_dir / "experiment_config.json", "w") as f:
            json.dump(expt_config.model_dump_json(indent=2), f)

        # Load inference query set
        query_set = InferenceQuerySet.from_json(query_json)

        # Perform MSA computation if selected
        #  update query_set with MSA paths
        if expt_runner.use_msa_server:
            logger.info("Using ColabFold MSA server for alignments.")
            query_set = preprocess_colabfold_msas(
                inference_query_set=query_set,
                compute_settings=expt_config.msa_computation_settings,
            )

            # Update the msa dataset config settings
            updated_dataset_config_kwargs = expt_config.dataset_config_kwargs.model_copy(
                update={"msa": colabfold_msa_settings}
            )
            expt_config = expt_config.model_copy(
                update={"dataset_config_kwargs": updated_dataset_config_kwargs}
            )
        else:
            expt_config.msa_computation_settings.cleanup_msa_dir = False

        # Preprocess template alignments and optionally template structures
        if expt_runner.use_templates:
            logger.info("Using templates for inference.")
            template_preprocessor = TemplatePreprocessor(
                input_set=query_set,
                config=expt_config.dataset_config_kwargs.template_preprocessor,
            )
            template_preprocessor()
        else:
            logger.info("Not using templates for inference.")

        # Run the forward pass
        expt_runner.setup()
        expt_runner.run(query_set)
        expt_runner.cleanup()

        logger.info("Inference completed successfully.")
