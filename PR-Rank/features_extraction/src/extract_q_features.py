import csv
import logging
from math import log
from pathlib import Path

import jsonlines
import numpy as np
from features_extraction.src.collection import Collection
from omegaconf import OmegaConf
from utils.text_utils import count_valid_tokens

logger = logging.getLogger(__name__)


def _query_idf(query_tokens, collection) -> list[float]:
    idfs = []

    dn = float(collection.dn)
    for token in query_tokens:
        df = collection.df.get(token, 0)
        idf = log(dn / (df + 1)) + 1
        idfs.append(idf)

    return idfs


def query_idf_sum(query_tokens, collection) -> float:
    idfs = _query_idf(query_tokens, collection)
    return sum(idfs)


def query_mean_idf(query_tokens, collection) -> float:
    idfs = _query_idf(query_tokens, collection)
    return float(np.mean(idfs))


def query_max_idf(query_tokens, collection) -> float:
    idfs = _query_idf(query_tokens, collection)
    return float(np.max(idfs))


def query_idf_sd(query_tokens, collection) -> float:
    idfs = _query_idf(query_tokens, collection)
    return float(np.std(idfs))


def _query_icf(query_tokens, collection) -> list[float]:
    icfs = []

    cn = float(collection.cn)
    for token in query_tokens:
        cf = collection.cf.get(token, 0)
        icf = log(cn / (cf + 1)) + 1
        icfs.append(icf)

    return icfs


def query_icf_sum(query_tokens, collection) -> float:
    icfs = _query_icf(query_tokens, collection)
    return sum(icfs)


def query_mean_icf(query_tokens, collection) -> float:
    icfs = _query_icf(query_tokens, collection)
    return float(np.mean(icfs))


def extract_q_features(query_tokens, collection) -> list[float]:
    query_length = sum(count_valid_tokens(query_tokens).values())

    features = [
        query_length,
        query_idf_sum(query_tokens, collection),
        query_mean_idf(query_tokens, collection),
        query_max_idf(query_tokens, collection),
        query_idf_sd(query_tokens, collection),
        query_icf_sum(query_tokens, collection),
        query_mean_icf(query_tokens, collection),
    ]
    return features


def run(config: OmegaConf):
    logger.info("Calculating Q features")
    query_features_file_path = Path(config.features.document_features_file_path)

    if not query_features_file_path.exists():
        collection = Collection.load(config.preprocess.collection_file_path)
        with (
            jsonlines.open(config.preprocess.query_details_file_path, "r") as reader,
            open(query_features_file_path, "w") as out_f,
        ):
            writer = csv.writer(out_f)
            for query_detail in reader:
                query_id = query_detail["query_id"]
                query = query_detail["tokens"]
                features = extract_q_features(query, collection)
                writer.writerow(query_id + features)
    else:
        logger.info("Q features already exist. Skipping...")

    logger.info("Calculating Q features completed")
