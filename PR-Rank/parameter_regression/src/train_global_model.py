import concurrent.futures
import logging
from pathlib import Path

from omegaconf import OmegaConf
from tqdm import tqdm
from utils.ranklib_utils import read_score, train_and_eval_model

logger = logging.getLogger(__name__)


def aggrigate_datasets(input_dir_path: Path, output_file_path: Path):
    with open(output_file_path, "w") as out_f:
        for dataset_dir_path in tqdm(Path(input_dir_path).glob("dataset_*")):
            for step in ["train", "valid", "test"]:
                dataset_file_path = dataset_dir_path / f"{step}.txt"
                if not dataset_file_path.exists():
                    continue
                with open(dataset_file_path, "r") as in_f:
                    for line in in_f:
                        out_f.write(line)


def run(config: OmegaConf):
    large_datasets_dir_path = Path(config.global_model.large_dataset_dir_path)
    for step in ["train", "valid", "test"]:
        input_dir_path = Path(config.ltr_datasets_dir_path) / step
        output_file_path = large_datasets_dir_path / f"{step}.txt"
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        aggrigate_datasets(input_dir_path, output_file_path)

    train_file_path = large_datasets_dir_path / "train.txt"
    validation_file_path = large_datasets_dir_path / "valid.txt"
    test_file_path = large_datasets_dir_path / "test.txt"
    model_dir_path = Path(config.global_model.global_model_dir_path)
    model_dir_path.mkdir(parents=True, exist_ok=True)

    iteration = config.global_model.iteration
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=config.global_model.parallel
    ) as executor:
        for i in range(iteration):
            model_file_path = model_dir_path / f"model_{i}.dat"
            idv_file_path = model_dir_path / f"eval_{i}.txt"

            executor.submit(
                train_and_eval_model,
                config.ranklib_jar_path,
                train_file_path,
                validation_file_path,
                model_file_path,
                test_file_path,
                idv_file_path,
                iteration=1,
            )

    best_score = float("-inf")
    best_model_index = -1

    for i in range(iteration):
        idv_file_path = model_dir_path / f"eval_{i}.txt"
        _, value = read_score(idv_file_path)
        logger.info(f"iteration: {i}, score: {value}")
        if value > best_score:
            best_score = value
            best_model_index = i

    for i in range(iteration):
        model_file_path = model_dir_path / f"model_{i}.dat"
        idv_file_path = model_dir_path / f"eval_{i}.txt"
        if i == best_model_index:
            logger.info(f"best iteration: {i}, score: {best_score}")
            # rename
            model_file_path.rename(model_dir_path / "model.dat")
            idv_file_path.rename(model_dir_path / "eval.txt")
        else:
            # delete file
            model_file_path.unlink()
            idv_file_path.unlink()
