import csv
import logging
from pathlib import Path

from omegaconf import OmegaConf
from utils.ranklib_utils import gather_parameters

logger = logging.getLogger(__name__)


def save_parameters_as_csv(model_weights: list[float], weights_dir_path: Path) -> None:
    with open(weights_dir_path, "w") as f:
        writer = csv.writer(f)
        for dataset_id, weight in model_weights.items():
            writer.writerow([dataset_id] + weight)


def run(config: OmegaConf):
    logger.info("Extracting model parameters")

    model_parameters_dir_path = Path(config.model_parameters_dir_path)

    if not model_parameters_dir_path.exists():
        model_parameters_dir_path.mkdir(parents=True)

        ltr_models_dir_path = Path(config.ltr_models_dir_path)

        for step in ["train", "valid", "test"]:
            group_dir_path = ltr_models_dir_path / step
            assert group_dir_path.exists()

            model_parameters = gather_parameters(group_dir_path)
            save_parameters_as_csv(
                model_parameters, model_parameters_dir_path / f"{step}.csv"
            )

    else:
        logger.info("Domain features already exist")

    logger.info("Extracting domain features completed")
