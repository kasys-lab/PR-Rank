import csv
import logging
from pathlib import Path
from typing import Callable

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
from utils.aolia_utils import FEATURES_INDEX
from utils.ranklib_utils import parse_ranklib_line

logger = logging.getLogger(__name__)


def extract_domain_features_by_group(
    group_dir_path: Path,
    extract_method: Callable[[Path], list[float]],
    features_index: list[int],
) -> dict[str, list[float]]:
    domain_feature_by_group: dict[str, list[float]] = {}

    for dataset_dir_path in tqdm(group_dir_path.glob("dataset_*")):
        dataset_name = dataset_dir_path.stem.split("_")[-1]
        domain_features = extract_method(dataset_dir_path, features_index)
        domain_feature_by_group[dataset_name] = domain_features

    return domain_feature_by_group


def calculate_query_features_from_dataset_file(
    dataset_file_path: Path,
    skip_condition: Callable[[int], bool] = lambda _: False,
) -> list[list[float]]:
    query_features: list[list[float]] = []
    instances_features_with_same_qid: list[list[float]] = []
    prev_qid = None

    with open(dataset_file_path, "r") as f:
        for line in f:
            relevance, qid, features, comment = parse_ranklib_line(line)
            if skip_condition(relevance):
                continue
            if prev_qid is None:
                prev_qid = qid
            if qid != prev_qid:
                query_features.append(
                    np.mean(instances_features_with_same_qid, axis=0).tolist()
                )
                prev_qid = qid
                instances_features_with_same_qid = []
            instances_features_with_same_qid.append(features)

    if instances_features_with_same_qid:
        query_features.append(
            np.mean(instances_features_with_same_qid, axis=0).tolist()
        )

    return query_features


def extract_domain_features_from_all_instances(
    domain_dir_path: Path, features_index: list[int]
) -> list[float]:
    all_query_features = []
    for step in ["train", "valid", "test"]:
        dataset_file_path = domain_dir_path / f"{step}.txt"
        if not dataset_file_path.exists():
            continue
        all_query_features.extend(
            calculate_query_features_from_dataset_file(dataset_file_path)
        )
    all_query_features = [
        [query_features[i] for i in features_index]
        for query_features in all_query_features
    ]

    domain_features = np.mean(all_query_features, axis=0).tolist()
    return domain_features


def extract_domain_features_from_positive_instances(
    domain_dir_path: Path, features_index: list[int]
) -> list[float]:
    all_query_features = []
    for step in ["train", "valid"]:
        dataset_file_path = domain_dir_path / f"{step}.txt"
        if not dataset_file_path.exists():
            continue
        all_query_features.extend(
            calculate_query_features_from_dataset_file(
                dataset_file_path, lambda rel: rel <= 0
            )
        )
    all_query_features = [
        [query_features[i] for i in features_index]
        for query_features in all_query_features
    ]

    domain_features = np.mean(all_query_features, axis=0).tolist()
    return domain_features


def save_domain_features(
    domain_features_by_group: dict[int, list[float]], domain_features_file_path: Path
) -> None:
    with open(domain_features_file_path, "w") as f:
        writer = csv.writer(f)
        for domain_name, domain_features in domain_features_by_group.items():
            writer.writerow([domain_name, *domain_features])


def run(config: OmegaConf):
    logger.info("Extracting domain features")

    EXTRACT_METHODS = {
        "from_all_instances": extract_domain_features_from_all_instances,
        "from_positive_instances": extract_domain_features_from_positive_instances,
    }

    features_index = []
    for feature_set_name in config.domain_feature_sets:
        features_index.extend(FEATURES_INDEX[feature_set_name])

    domain_features_dir_path = Path(config.domain_features_dir_path)

    if not domain_features_dir_path.exists():
        domain_features_dir_path.mkdir(parents=True)
        domains_dir_path = Path(config.domains_dir_path)

        for step in ["train", "valid", "test"]:
            group_dir_path = domains_dir_path / step
            assert group_dir_path.exists()

            domain_features_by_group = extract_domain_features_by_group(
                group_dir_path, EXTRACT_METHODS[config.extract_method], features_index
            )

            domain_features_file_path = domain_features_dir_path / f"{step}.csv"
            save_domain_features(domain_features_by_group, domain_features_file_path)
    else:
        logger.info("Domain features already exist")

    logger.info("Extracting domain features completed")
