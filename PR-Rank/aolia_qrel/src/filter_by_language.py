import logging
from collections import defaultdict
from pathlib import Path

import jsonlines
from ir_datasets.datasets.base import Dataset
from omegaconf import OmegaConf
from tqdm import tqdm
from utils.file_utils import load_lines_as_set, save_iter_to_lines, save_pickle

logger = logging.getLogger(__name__)


def is_target_language(language: str, target_labels: list) -> bool:
    return language in target_labels


def filter_documents(input_file: Path, target_labels: list) -> set:
    filtered_doc_ids = set()

    with jsonlines.open(input_file, "r") as reader:
        for doc in reader:
            if is_target_language(
                doc["text_language"], target_labels
            ) and is_target_language(doc["title_language"], target_labels):
                filtered_doc_ids.add(doc["doc_id"])

    return filtered_doc_ids


def filter_queries(input_file: Path, target_labels: list) -> set:
    filtered_query_ids = set()

    with jsonlines.open(input_file, "r") as reader:
        for query in reader:
            if is_target_language(query["text_language"], target_labels):
                filtered_query_ids.add(query["query_id"])

    return filtered_query_ids


def filter_relevant_docs(
    dataset: Dataset,
    filtered_doc_ids: set[str],
    filtered_query_ids: set[str],
) -> dict[str, set[str]]:
    if not (filtered_doc_ids and filtered_query_ids):
        raise ValueError(
            "filtered_doc_ids and filtered_query_ids must be non-empty sets."
        )

    relevant_docs = defaultdict(set)
    print("creating relevant docs...")

    for qrel in tqdm(dataset.qrels_iter(), total=dataset.qrels_count()):
        if not (
            qrel.query_id in filtered_query_ids and qrel.doc_id in filtered_doc_ids
        ):
            continue

        relevant_docs[qrel.query_id].add(qrel.doc_id)

    return relevant_docs


def run(dataset: Dataset, config: OmegaConf):
    logger.info("Filtering documents and queries")

    # filter documents
    filtered_doc_ids_file_path = Path(config.filter.filtered_doc_ids_file_path)
    if not filtered_doc_ids_file_path.exists():
        filtered_doc_ids = filter_documents(
            config.filter.document_languages_file_path,
            config.filter.target_labels,
        )
        save_iter_to_lines(filtered_doc_ids, filtered_doc_ids_file_path)
    else:
        logger.info("Passing document filtering as the file already exists")

    # filter queries
    filtered_query_ids_file_path = Path(config.filter.filtered_query_ids_file_path)
    if not filtered_query_ids_file_path.exists():
        filtered_query_ids = filter_queries(
            config.filter.query_languages_file_path,
            config.filter.target_labels,
        )
        save_iter_to_lines(filtered_query_ids, filtered_query_ids_file_path)
    else:
        logger.info("Passing query filtering as the file already exists")

    # filter qrels
    filtered_relevant_docs_file_path = Path(
        config.filter.filtered_relevant_docs_file_path
    )
    if not filtered_relevant_docs_file_path.exists():
        filtered_doc_ids = load_lines_as_set(filtered_doc_ids_file_path)
        filtered_query_ids = load_lines_as_set(filtered_query_ids_file_path)
        filtered_relevant_docs = filter_relevant_docs(
            dataset,
            filtered_doc_ids,
            filtered_query_ids,
        )
        save_pickle(filtered_relevant_docs, filtered_relevant_docs_file_path)
    else:
        logger.info("Passing qrel filtering as the file already exists")

    logger.info("Documents and queries filtered")
