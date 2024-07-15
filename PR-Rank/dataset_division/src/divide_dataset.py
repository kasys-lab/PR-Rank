import csv
import json
import logging
import random
from pathlib import Path

from omegaconf import OmegaConf
from tqdm import tqdm
from utils.aolia_utils import FEATURES_INDEX
from utils.ranklib_utils import convert_features_to_ranklib_format

logger = logging.getLogger(__name__)


# Determine which dataset (train, valid, test) each query belongs to for each domain
def split_query(
    cluster2qid: dict[int, list[int]],
    train_rate: float = 0.7,
    valid_rate: float = 0.15,
    seed: int = 42,
) -> dict[int, tuple[int, str]]:
    random.seed(seed)
    qid2location = {}

    for cluster_id, qids in cluster2qid.items():
        random.shuffle(qids)

        train_size = int(len(qids) * train_rate)
        valid_size = int(len(qids) * valid_rate)

        # Ensure there's at least one sample in train
        train_size = max(1, train_size)

        # Adjust valid size if we've taken an extra sample for training
        if len(qids) - train_size < valid_size:
            valid_size = len(qids) - train_size

        steps = {
            "train": qids[:train_size],
            "valid": qids[train_size : train_size + valid_size],
            "test": qids[train_size + valid_size :],
        }

        # Add (cluster_id, step) to the dictionary for each step
        for step, qids in steps.items():
            for qid in qids:
                qid2location[qid] = (cluster_id, step)

    return qid2location


def assign_step_to_domains(
    num_of_domains: int,
    train_rate: float,
    valid_rate: float,
    seed: int = 42,
) -> dict[str, str]:
    random.seed(seed)
    domain2step = {}

    num_train = int(num_of_domains * train_rate)
    num_valid = int(num_of_domains * valid_rate)
    num_test = num_of_domains - num_train - num_valid

    steps = ["train"] * num_train + ["valid"] * num_valid + ["test"] * num_test

    random.shuffle(steps)

    for domain_id, step in enumerate(steps):
        domain2step[str(domain_id)] = step

    return domain2step


# divide the entire dataset into domains
def divide_dataset(
    feature_file_path: Path,
    domain2step: dict[str, str],
    qid2location: dict[str, int],
    domains_dir_path: Path,
    features_index: list[int],
):
    # assign each line to a domain based on the query ID
    with open(feature_file_path, "r") as in_f:
        reader = csv.reader(in_f)
        for line in tqdm(reader):
            relevance = int(line[0])
            query_id = line[1]
            doc_id = line[2]
            futures = [float(x) for x in line[3:]]
            futures = [futures[i] for i in features_index]
            cluster_id, step = qid2location.get(query_id)
            meta_step = domain2step.get(cluster_id)

            dataset_dir_path = (
                Path(domains_dir_path) / meta_step / f"dataset_{cluster_id}"
            )
            if not dataset_dir_path.exists():
                dataset_dir_path.mkdir()

            with open(dataset_dir_path / f"{step}.txt", "a") as out_f:
                out_f.write(
                    convert_features_to_ranklib_format(
                        int(query_id, 16), doc_id, relevance, futures
                    )
                    + "\n"
                )
    return


def extract_qid(line):
    return int(line.split(" ")[1].split(":")[1])


def sort_file_by_qid(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    lines_with_qid = [(extract_qid(line), line) for line in lines]
    sorted_lines_with_qid = sorted(lines_with_qid, key=lambda x: x[0])
    sorted_lines = [line for _, line in sorted_lines_with_qid]

    with open(file_path, "w") as file:
        for line in sorted_lines:
            file.write(line)


def sort_files_by_qid(domains_dir_path):
    for dataset_file_path in domains_dir_path.glob("*/*/*.txt"):
        try:
            sort_file_by_qid(dataset_file_path)
        except ValueError:
            print(f"{dataset_file_path}...")
            continue


def run(config: OmegaConf):
    logger.info("dividing the dataset into domains")
    domains_dir_path = Path(config.cluster.domains_dir_path)
    ltr_datasets_dir_path = Path(config.cluster.ltr_datasets_dir_path)

    if not domains_dir_path.exists():
        domains_dir_path.mkdir(parents=True)
        ltr_datasets_dir_path.mkdir(parents=True)
        for step in ["train", "valid", "test"]:
            (domains_dir_path / step).mkdir()
            (ltr_datasets_dir_path / step).mkdir()

        with open(config.cluster.cluster2qids_file_path, "r") as f:
            cluster2qids = json.load(f)

        domain2step = assign_step_to_domains(
            config.cluster.number_of_domains, 0.7, 0.15
        )
        with open(config.cluster.domain2step_file_path, "w") as f:
            json.dump(domain2step, f)

        # domain2step = json.load((experiment_dir_path / "utils" / "domain2step.json").open())

        qid2location = split_query(cluster2qids)

        # for ltr dataset
        divide_dataset(
            config.data.features_file_path,
            domain2step,
            qid2location,
            config.cluster.ltr_datasets_dir_path,
            FEATURES_INDEX["LtR"],
        )
        sort_files_by_qid(domains_dir_path)

        # for pr-rank dataset
        divide_dataset(
            config.data.features_file_path,
            domain2step,
            qid2location,
            domains_dir_path,
            FEATURES_INDEX["Q"] + FEATURES_INDEX["D"] + FEATURES_INDEX["Q-D"],
        )
        sort_files_by_qid(domains_dir_path)
    else:
        logger.info("domains already divided")

    logger.info("dividing the dataset into domains completed")
