import subprocess
from pathlib import Path
from typing import Optional


def convert_features_to_ranklib_format(
    query_id: str, doc_id: str, relevance: int, futures: list[float]
):
    indexed_features = convert_list_to_indexed_string(futures)
    ranklib_line = f"{relevance} qid:{query_id} {indexed_features} # doc_id={doc_id}"
    return ranklib_line


def parse_ranklib_line(line) -> tuple[int, str, list[float], str]:
    parts = line.strip().split(" ", 2)
    relevance = int(parts[0])
    qid = str(parts[1].split(":")[1])
    features_and_comment = parts[2].split("#")

    features = convert_indexed_string_to_list(features_and_comment[0])

    comment = features_and_comment[1].strip() if len(features_and_comment) > 1 else ""

    return relevance, qid, features, comment


# example
# "1: 0.1, 2: 0.2, 3: 0.3" -> [0.1, 0.2, 0.3]
def convert_indexed_string_to_list(weight_line: str) -> list[float]:
    weight_list = []
    for each_weight in weight_line.split():
        _feature_id, value = each_weight.split(":")
        weight_list.append(float(value))
    return weight_list


# example
# [0.1, 0.2, 0.3] -> "1: 0.1, 2: 0.2, 3: 0.3"
def convert_list_to_indexed_string(weight_list: list[float]) -> str:
    weight_line = " ".join(
        f"{idx}:{weight}" for idx, weight in enumerate(weight_list, 1)
    )
    return weight_line


def train_model(
    ranklib_jar_path: Path,
    train_file_path: Path,
    validation_file_path: Path,
    model_file_path: Path,
    iteration: int = 5,
):
    cmd_options = {
        "ranker": "4",  # Coordinate Ascent
        "train": str(train_file_path),
        "reg": "0.1",  # チューニングが必要？
        "metric2t": "NDCG@10",
        "norm": "zscore",
        "r": str(iteration),
        "save": str(model_file_path),
    }

    if validation_file_path is not None:
        cmd_options["validate"] = str(validation_file_path)

    cmd = ["java", "-jar", str(ranklib_jar_path)]
    for option, value in cmd_options.items():
        cmd.extend([f"-{option}", value])

    cmd.append("-silent")

    subprocess.run(cmd)
    return


def eval_model(
    ranklib_jar_path: Path,
    test_file_path: Path,
    model_file_path: Path,
    idv_file_path: Path,
    metrics: str = "NDCG@10",
    ranker_id: int = 4,
):
    cmd_options = {
        "ranker": str(ranker_id),  # 4: Coordinate Ascent
        "load": str(model_file_path),
        "test": str(test_file_path),
        "idv": str(idv_file_path),
        "metric2T": metrics,
        "norm": "zscore",
    }

    cmd = ["java", "-jar", str(ranklib_jar_path)]
    for option, value in cmd_options.items():
        cmd.extend([f"-{option}", value])

    cmd.append("-silent")

    subprocess.run(cmd)
    return


def rank_by_model(
    ranklib_jar_path: Path,
    test_file_path: Path,
    model_file_path: Path,
    score_file_path: Path,
    ranker_id: int = 4,  # 4: Coordinate Ascent
):
    cmd_options = {
        "ranker": str(ranker_id),
        "load": str(model_file_path),
        "rank": str(test_file_path),
        "score": str(score_file_path),
        "norm": "zscore",
    }

    cmd = ["java", "-jar", str(ranklib_jar_path)]
    for option, value in cmd_options.items():
        cmd.extend([f"-{option}", value])

    cmd.append("-silent")

    subprocess.run(cmd)
    return


def train_and_eval_model(
    ranklib_jar_path: Path,
    train_file_path: Path,
    validation_file_path: Optional[Path],
    ltr_model_file_path: Path,
    test_file_path: Optional[Path],
    idv_file_path: Path,
    iteration: int = 5,
) -> None:
    train_model(
        ranklib_jar_path,
        train_file_path,
        validation_file_path,
        ltr_model_file_path,
        iteration,
    )

    if test_file_path is not None:
        eval_model(ranklib_jar_path, test_file_path, ltr_model_file_path, idv_file_path)


def read_score(idv_file: Path) -> tuple[str, float]:
    with open(idv_file, "r") as f:
        for line in f:
            pass
        last_line = line
    metric, _, value = last_line.split()
    return metric, float(value)


def gather_parameters(datasets_dir_path: Path) -> dict[str, list[float]]:
    model_weights = {}
    for dataset_file_path in datasets_dir_path.glob("dataset_*"):
        dataset_id = dataset_file_path.stem.split("_")[-1]
        with open(dataset_file_path / "model.dat", "r") as f:
            for line in f:
                pass
            last_line = line
        weight = convert_indexed_string_to_list(last_line)
        model_weights[dataset_id] = weight
    return model_weights


def save_ltr_model(ltr_model_weight: str, model_dir_path: Path):
    weight_line = convert_list_to_indexed_string(ltr_model_weight)

    with open(model_dir_path, "w") as f:
        f.write("## Coordinate Ascent\n")
        f.write(weight_line)
