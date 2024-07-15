import logging
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from utils.ranklib_utils import gather_parameters

logger = logging.getLogger(__name__)


def calculate_parameter_variance(datasets_dir_path) -> dict:
    model_parameters_matrix = []

    for step in ["train", "valid", "test"]:
        model_parameters = gather_parameters(Path(datasets_dir_path) / step)
        model_parameters_matrix.extend([v for _, v in model_parameters.items()])

    parameter_variance = np.sum(np.var(model_parameters_matrix, axis=0).tolist())
    return parameter_variance


def run(config: OmegaConf):
    logger.info("Evaluation dataset division")

    eval_file_path = Path(config.eval.eval_file_path)
    if not eval_file_path.exists():
        parameter_variance = calculate_parameter_variance(config.train.ltr_models_dir_path)
        with open(eval_file_path, "w") as f:
            f.write(f"Parameter variance: {parameter_variance}\n")
    else:
        logger.info("LTR models already exist. Skipping...")

    logger.info("Training LTR models is done.")
