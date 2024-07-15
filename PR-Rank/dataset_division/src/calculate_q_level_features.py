import csv
import logging
from pathlib import Path

from omegaconf import OmegaConf
from tqdm import tqdm

logger = logging.getLogger(__name__)


def aggregate_features(qd_level_feature_file_path: str, output_file_path: str):
    features_by_query = {}

    with open(qd_level_feature_file_path, "r") as in_f:
        reader = csv.reader(in_f)

        for row in tqdm(reader):
            query_id = row[1]
            features = [float(f) for f in row[3:]]

            if query_id not in features_by_query:
                features_by_query[query_id] = {"cnt": 1, "features": features}
                continue

            features_by_query[query_id]["cnt"] += 1
            features_by_query[query_id]["features"] = [
                f1 + f2
                for f1, f2 in zip(features_by_query[query_id]["features"], features)
            ]

    with open(output_file_path, "w") as out_f:
        writer = csv.writer(out_f)
        for query_id, data in tqdm(features_by_query.items()):
            features = data["features"]
            cnt = data["cnt"]
            features = [f / cnt for f in features]
            writer.writerow([query_id] + features)


def run(config: OmegaConf):
    logger.info("Aggregating q-d level features to query level features")

    q_level_features_file_path = Path(config.data.q_level_features_file_path)

    if not q_level_features_file_path.exists():
        aggregate_features(config.data.features_file_path, q_level_features_file_path)
    else:
        logger.info("q-level features already exist. Skip the process.")

    logger.info("Aggregating q-d level features to query level features completed")
