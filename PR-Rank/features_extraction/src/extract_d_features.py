import csv
import json
import logging
from math import log
from pathlib import Path

import jsonlines
from ir_datasets.datasets.base import Dataset
from omegaconf import OmegaConf
from tqdm import tqdm
from utils.file_utils import load_pickle
from utils.text_utils import count_valid_tokens

logger = logging.getLogger(__name__)


def dlen(d):
    return sum(d.values())


def log_dlen(d):
    return log(dlen(d) + 1)


def domain_freq(domain: str, dc: dict[str, int]):
    return dc.get(domain, 1)


def domain_length(domain: str):
    return len(domain)


def num_of_slash(url: str):
    return url.count("/") - 2


# ARI (Automated Readability Index)
def calc_ari_score(text: list[str], sentences: int) -> float:
    characters = sum([len(word) for word in text])
    words = max(len(text), 1)
    sentences = max(sentences, 1)
    return 4.71 * (characters / words) + 0.5 * (words / sentences) - 21.43


def extract_d_features(document, document_detail, domain_counter, clf, tf_idf_pipe):
    text_tokens = [
        lemma_pos_pair[0] for lemma_pos_pair in document_detail["text_tokens"]
    ]
    title_tokens = [
        lemma_pos_pair[0] for lemma_pos_pair in document_detail["text_tokens"]
    ]
    sentence_count = document_detail["sentence_count"]
    domain = document_detail["second_level_domain"]

    x = tf_idf_pipe.transform(text_tokens + title_tokens)
    topic_proba = clf.predict_proba(x)[0].tolist()

    text = count_valid_tokens(text_tokens)
    title = count_valid_tokens(title_tokens)

    features = [
        dlen(text),
        dlen(title),
        log_dlen(text),
        log_dlen(title),
        calc_ari_score(text_tokens, sentence_count),
        sentence_count,
        *topic_proba,
        domain_freq(domain, domain_counter),
        domain_length(domain),
        num_of_slash(document.url),
    ]

    return features


def run(dataset: Dataset, config: OmegaConf):
    logger.info("Extracting D features")
    document_features_file_path = Path(config.features.document_features_file_path)

    if not document_features_file_path.exists():
        docs_store = dataset.docs_store()
        domain_counter = json.load(open(config.preprocess.sld_counter_file_path, "r"))
        clf = load_pickle(config.topic_estimator.model_file_path)
        tf_idf_pipe = load_pickle(config.topic_estimator.vectorizer_file_path)

        with (
            jsonlines.open(config.preprocess.document_details_file_path, "r") as reader,
            open(document_features_file_path, "w") as out_f,
        ):
            writer = csv.writer(out_f)
            for document_detail in tqdm(reader):
                doc_id = document_detail["doc_id"]
                doc = docs_store.get(doc_id)
                features = extract_d_features(
                    doc, document_detail, domain_counter, clf, tf_idf_pipe
                )
                writer.writerow([doc_id] + features)
    else:
        logger.info("Document features already exist. Skipping...")

    logger.info("Extracting D features completed")
