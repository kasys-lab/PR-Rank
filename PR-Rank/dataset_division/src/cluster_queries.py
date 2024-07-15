import csv
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

from omegaconf import OmegaConf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from utils.aolia_utils import FEATURES_INDEX

logger = logging.getLogger(__name__)


def load_q_level_domain_features(
    q_level_domain_features_file_path: Path, features_to_use: list[str]
) -> tuple[list[str], list[list[float]]]:
    query_ids, feature_matrix = [], []

    with open(q_level_domain_features_file_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            query_id = row[0]
            features = row[1:]

            query_ids.append(query_id)
            feature_matrix.append(
                [float(features[i]) for i in features_to_use if i < len(features)]
            )

    return query_ids, feature_matrix


def normalize(feature_matrix: list[list[float]]) -> list[list[float]]:
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)
    return scaled_features


def clustering(scaled_features: list[list[float]], n_clusters) -> list[int]:
    if len(scaled_features[0]) == 1:
        print("1-dim clustering")
        scaled_features_with_index = [
            (i, scaled_feature[0]) for i, scaled_feature in enumerate(scaled_features)
        ]
        scaled_features_with_index.sort(key=lambda x: x[1])

        elements_per_label = len(scaled_features_with_index) // n_clusters
        remaining_elements = len(scaled_features_with_index) % n_clusters

        label_with_index = []
        index_iterator = iter([i for i, _ in scaled_features_with_index])
        for label in range(n_clusters):
            label_with_index.extend(
                [(next(index_iterator), label) for _ in range(elements_per_label)]
            )
            if label < remaining_elements:
                label_with_index.append((next(index_iterator), label))
        assert len(label_with_index) == len(scaled_features_with_index)
        labels = [label for _, label in sorted(label_with_index, key=lambda x: x[0])]

    else:
        model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42, verbose=1)
        model.fit(scaled_features)
        labels = model.labels_
    return labels


def random_clustering(
    scaled_feature_matrix: list[list[float]], n_clusters
) -> list[int]:
    labels = [
        random.randint(0, n_clusters - 1) for _ in range(len(scaled_feature_matrix))
    ]
    return labels


def output_result(
    labels: list[int], query_ids: list[str]
) -> tuple[dict[str, str], dict[str, list[str]]]:
    query2domain = {}
    domain2queries = defaultdict(list)

    for label, query_id in zip(labels, query_ids):
        domain_id = str(label)
        query2domain[query_id] = domain_id
        domain2queries[domain_id].append(query_id)

    return query2domain, domain2queries


def run(config: OmegaConf):
    logger.info("Clustering queries...")

    qid2cluster_file_path = Path(config.cluster.qid2cluster_file_path)

    if not qid2cluster_file_path.exists():
        qid2cluster_file_path.parent.mkdir(parents=True)

        features_to_use = []
        for feature_set in config.cluster.feature_sets:
            features_to_use.extend(FEATURES_INDEX[feature_set])

        query_ids, feature_matrix = load_q_level_domain_features(
            config.data.q_level_features_file_path, features_to_use
        )

        scaled_features = normalize(feature_matrix)

        match config.cluster.method:
            case "random":
                logger.info("Random clustering")
                labels = random_clustering(
                    scaled_features, config.cluster.number_of_domains
                )
            case "kmeans":
                logger.info("Kmeans clustering")
                labels = clustering(scaled_features, config.cluster.number_of_domains)

        qid2cluster, cluster2qids = output_result(labels, query_ids)

        json.dump(qid2cluster, open(config.cluster.qid2cluster_file_path, "w"))
        json.dump(cluster2qids, open(config.cluster.cluster2qids_file_path, "w"))
    else:
        logger.info("Clustering queries file already exists")

    logger.info("Clustering queries done")
