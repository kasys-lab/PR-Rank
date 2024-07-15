import csv
import logging
from math import log
from pathlib import Path

import jsonlines
from features_extraction.src.collection import Collection
from omegaconf import OmegaConf
from utils.file_utils import sort_jsonl
from utils.text_utils import count_valid_tokens

logger = logging.getLogger(__name__)


def tf_sum(q, d, c):
    """
    ID 1 for OHSUMED
    """
    result = 0.0
    for w in set(q) & set(d):
        result += d[w]
    return result


def log_tf_sum(q, d, c):
    """
    ID 2 for OHSUMED
    """
    result = 0.0
    for w in set(q) & set(d):
        result += log(d[w] + 1)
    return result


def norm_tf_sum(q, d, c):
    """
    ID 3 for OHSUMED
    """
    result = 0.0
    dlen = sum(d.values())
    for w in set(q) & set(d):
        result += d[w]
    if dlen > 0:
        result /= dlen
    return result


def log_norm_tf_sum(q, d, c):
    """
    ID 4 for OHSUMED
    """
    result = 0.0
    dlen = sum(d.values())
    for w in set(q) & set(d):
        result += log(float(d[w]) / dlen + 1)
    return result


def idf_sum(q, d, c):
    """
    ID 5 for OHSUMED
    """
    result = 0.0
    for w in set(q) & set(d):
        result += log(float(c.dn) / c.df[w])
    return result


def log_idf_sum(q, d, c):
    """
    ID 6 for OHSUMED
    """
    result = 0.0
    for w in set(q) & set(d):
        logval = log(float(c.dn) / c.df[w])
        if logval > 0:
            result += log(logval)
    return result


def icf_sum(q, d, c):
    """
    ID 7 for OHSUMED
    """
    result = 0.0
    for w in set(q) & set(d):
        result += log(float(c.dn) / c.cf[w] + 1)
    return result


def log_tfidf_sum(q, d, c):
    """
    ID 8 for OHSUMED
    """
    result = 0.0
    dlen = sum(d.values())
    if dlen == 0:
        return result
    for w in set(q) & set(d):
        result += log(float(d[w]) / dlen * log(float(c.dn) / c.df[w] + 1))
    return result


def tfidf_sum(q, d, c):
    """
    ID 9 for OHSUMED
    """
    result = 0.0
    for w in set(q) & set(d):
        result += d[w] * log(float(c.dn) / c.df[w])
    return result


def tf_in_idf_sum(q, d, c):
    """
    ID 10 for OHSUMED
    """
    result = 0.0
    dlen = sum(d.values())
    if dlen == 0:
        return result
    for w in set(q) & set(d):
        result += log(float(d[w]) / dlen * float(c.dn) / c.cf[w] + 1)
    return result


def _bm25_idf(w, c):
    return log((c.dn - c.df[w] + 0.5) / (c.df[w] + 0.5))


def bm25(q, d, c, k1=2.5, b=0.8):
    """
    ID 11 for OHSUMED
    """
    result = 0.0
    dlen = sum(d.values())
    if dlen == 0:
        return result
    for w in set(q) & set(d):
        result += (
            _bm25_idf(w, c)
            * d[w]
            * (k1 + 1)
            / (d[w] + k1 * (1 - b + b * dlen / c.avgdlen))
        )
    return result


def log_bm25(q, d, c, k1=2.5, b=0.8):
    """
    ID 12 for OHSUMED (+1 for log)
    """
    bm = bm25(q, d, c, k1, b)
    if bm > 0:
        return log(bm + 1.0)
    else:
        return 0.0


def _lm_pwc(w, c):
    """
    Add 1 for smoothing
    """
    return float(c.cf.get(w, 0.0) + 1.0) / (c.cn + len(c.df))


def lm_dir(q, d, c, mu=50.0):
    """
    ID 13 for OHSUMED
    """
    result = 0.0
    dlen = sum(d.values())
    if dlen == 0:
        return result
    qlen = sum(q.values())
    alpha = mu / (dlen + mu)
    for w in set(q):
        pwc = _lm_pwc(w, c)
        result += log(pwc)
        if w in d:
            pswd = (d[w] + mu * pwc) / (dlen + mu)
            result += log(pswd / (alpha * pwc))
    result += qlen * log(alpha)
    return result


def lm_jm(q, d, c, Lambda=0.5):
    """
    ID 14 for OHSUMED
    """
    result = 0.0
    dlen = sum(d.values())
    if dlen == 0:
        return result
    qlen = sum(q.values())
    for w in set(q):
        pwc = _lm_pwc(w, c)
        result += log(pwc)
        if w in d:
            pswd = (1 - Lambda) * d[w] / dlen + Lambda * pwc
            result += log(pswd / (Lambda * pwc))
    result += qlen * log(Lambda)
    return result


def lm_abs(q, d, c, delta=0.5):
    """
    ID 15 for OHSUMED
    """
    result = 0.0
    dlen = sum(d.values())
    if dlen == 0:
        return result
    qlen = sum(q.values())
    alpha = delta * len(d) / dlen
    for w in set(q):
        pwc = _lm_pwc(w, c)
        result += log(pwc)
        if w in d:
            pswd = max([0.0, d[w] - delta]) / dlen + alpha * pwc
            result += log(pswd / (alpha * pwc))
    result += qlen * log(alpha)
    return result


QD_FEATURES = [
    tf_sum,
    log_tf_sum,
    norm_tf_sum,
    log_norm_tf_sum,
    idf_sum,
    log_idf_sum,
    icf_sum,
    log_tfidf_sum,
    tfidf_sum,
    tf_in_idf_sum,
    bm25,
    log_bm25,
    lm_dir,
    lm_jm,
    lm_abs,
]


def make_query_features_dict(query_features_file_path: str):
    query_features_dict = {}
    with open(query_features_file_path, "r") as query_features_csv:
        query_features = csv.reader(query_features_csv)
        for row in query_features:
            query_id = row[0]
            features = list(map(float, row[1:]))
            query_features_dict[query_id] = features
    return query_features_dict


def make_query_details_dict(query_details_file_path: str):
    query_details_dict = {}
    with jsonlines.open(query_details_file_path, "r") as query_details:
        for query_detail in query_details:
            query_id = query_detail["query_id"]
            query = query_detail["tokens"]
            query_details_dict[query_id] = query
    return query_details_dict


def extract_qd_features(doc_detail, query_detail, collection):
    text = [lemma_pos_pair[0] for lemma_pos_pair in doc_detail["text_tokens"]]
    title = [lemma_pos_pair[0] for lemma_pos_pair in doc_detail["title_tokens"]]

    fields = [title, text]
    q = count_valid_tokens(query_detail)

    features = []
    for field in fields:
        d = count_valid_tokens(field)
        for qd_feature in QD_FEATURES:
            features.append(qd_feature(q, d, collection))

    return features


def process_files(
    sorted_relevance_judgement_file: str,
    document_details_file: str,
    document_features_file: str,
    query_details_dict: dict[str, list[str]],
    query_features_dict: dict[str, list[float]],
    query_document_features_file: str,
    features_file: str,
    collection: Collection,
):
    with (
        jsonlines.open(sorted_relevance_judgement_file, "r") as rel_judgements,
        jsonlines.open(document_details_file, "r") as doc_details,
        open(document_features_file, "r") as doc_features_csv,
        open(query_document_features_file, "w") as qd_features_csv,
        open(features_file, "w") as features_csv,
    ):
        qd_features_writer = csv.writer(qd_features_csv)
        features_writer = csv.writer(features_csv)

        rel_judgements = iter(rel_judgements)
        doc_details = iter(doc_details)
        doc_features = iter(csv.reader(doc_features_csv))

        rel_judgement = next(rel_judgements, None)
        doc_detail = next(doc_details, None)
        doc_feature = next(doc_features, None)

        while rel_judgement:
            current_doc_id = rel_judgement["doc_id"]

            if doc_detail and int(doc_detail["doc_id"], 16) < int(current_doc_id, 16):
                doc_detail = next(doc_details, None)
            if doc_feature and int(doc_feature[0], 16) < int(current_doc_id, 16):
                doc_feature = next(doc_features, None)

            if (
                doc_detail
                and int(doc_detail["doc_id"], 16) == int(current_doc_id, 16)
                and doc_feature
                and int(doc_feature[0], 16) == int(current_doc_id, 16)
            ):
                query_id = rel_judgement["query_id"]
                query_detail = query_details_dict[rel_judgement["query_id"]]
                query_features = query_features_dict[rel_judgement["query_id"]]

                qd_features = extract_qd_features(doc_detail, query_detail, collection)
                qd_features_writer.writerow([query_id, current_doc_id] + qd_features)

                relevance = rel_judgement["relevance"]
                features = query_features + doc_feature[1:] + qd_features
                features_writer.writerow(
                    [relevance, query_id, current_doc_id] + features
                )

            rel_judgement = next(rel_judgements, None)


def run(config: OmegaConf):
    logger.info("Extracting Q-D features")

    # sort relevance judgement file for efficient processing
    sorted_relevance_judgement_file_path = Path(
        config.data.sorted_relevance_judgement_file_path
    )
    if not sorted_relevance_judgement_file_path.exists():
        logger.info("Sorting relevance judgement file...")
        sort_jsonl(
            config.data.relevance_judgement_file_path,
            sorted_relevance_judgement_file_path,
            sort_key=lambda x: int(x["doc_id"], 16),
        )

    # extract Q-D features
    query_document_features_file_path = Path(
        config.features.query_document_features_file_path
    )
    if not query_document_features_file_path.exists():
        collection = Collection.load(open(config.preprocess.collection_file_path))
        query_features_dict = make_query_features_dict(
            config.features.query_features_file_path
        )
        query_details_dict = make_query_details_dict(
            config.preprocess.query_details_file_path
        )
        process_files(
            sorted_relevance_judgement_file_path,
            config.preprocess.document_details_file_path,
            config.features.document_features_file_path,
            query_details_dict,
            query_features_dict,
            config.features.query_document_features_file_path,
            config.features.features_file_path,
            collection,
        )
    else:
        logger.info("Q-D features already exist. Skipping...")

    logger.info("Extracting Q-D features completed")
