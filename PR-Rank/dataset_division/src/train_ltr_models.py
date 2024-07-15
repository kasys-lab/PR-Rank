import concurrent.futures
import logging
from pathlib import Path

from omegaconf import OmegaConf
from utils.ranklib_utils import train_and_eval_model

logger = logging.getLogger(__name__)


def train_and_eval_models_in_parallel(
    ranklib_jar_path: str,
    datasets_dir_path: str,
    ltr_models_dir_path: str,
    jobs: int = 4,
) -> None:
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as executor:
        for dataset_dir_path in Path(datasets_dir_path).glob("*/dataset_*"):
            step = dataset_dir_path.parent.name
            dataset_dir_name = dataset_dir_path.name

            train_file_path = dataset_dir_path / "train.txt"

            validation_file_path = dataset_dir_path / "valid.txt"
            if not validation_file_path.exists():
                validation_file_path = None

            test_file_path = dataset_dir_path / "test.txt"
            if not test_file_path.exists():
                logger.info(f"{test_file_path} does not exist.")
                test_file_path = None

            ltr_model_dir_path = Path(ltr_models_dir_path) / step / dataset_dir_name
            if ltr_model_dir_path.exists():
                print(f"Skip: {ltr_model_dir_path}")
                continue

            ltr_model_dir_path.mkdir(parents=True, exist_ok=False)

            model_file_path = ltr_model_dir_path / "model.dat"
            idv_file_path = ltr_model_dir_path / "eval.txt"

            executor.submit(
                train_and_eval_model,
                ranklib_jar_path,
                train_file_path,
                validation_file_path,
                model_file_path,
                test_file_path,
                idv_file_path,
                iteration=16,
            )

        return


def run(config: OmegaConf):
    logger.info("Training LTR models")
    ltr_models_dir_path = Path(config.train.ltr_models_dir_path)

    if not ltr_models_dir_path.exists():
        ltr_models_dir_path.mkdir(parents=True)
        for step in ["train", "valid", "test"]:
            (ltr_models_dir_path / step).mkdir(parents=True, exist_ok=False)

        train_and_eval_models_in_parallel(
            config.data.ranklib_jar_path,
            config.cluster.domains_dir_path,
            ltr_models_dir_path,
            jobs=config.train.jobs,
        )
    else:
        logger.info("LTR models already exist. Skipping...")

    logger.info("Training LTR models is done.")
