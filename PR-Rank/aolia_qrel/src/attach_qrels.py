import logging
from collections import namedtuple
from pathlib import Path

import jsonlines
from omegaconf import OmegaConf
from tqdm import tqdm
from utils.file_utils import load_pickle

logger = logging.getLogger(__name__)

RelevanceJudgement = namedtuple(
    "RelevanceJudgement", ["query_id", "doc_id", "relevance"]
)


def make_relevance_judgements(
    query_id: str,
    ranked_doc_ids: list[str],
    positive_doc_ids: set[str],
    negative_example_limit: int,
) -> list[RelevanceJudgement]:
    relevance_judgements = [
        RelevanceJudgement(query_id, doc_id, 1) for doc_id in positive_doc_ids
    ]

    negative_doc_cnt = 0
    for doc_id in ranked_doc_ids:
        if doc_id in positive_doc_ids:
            continue

        relevance_judgements.append(RelevanceJudgement(query_id, doc_id, 0))
        negative_doc_cnt += 1
        if negative_doc_cnt >= negative_example_limit:
            break

    return relevance_judgements


def save_relevance_judgements(
    ranking_file_path: Path,
    relevance_judgement_file_path: Path,
    relevant_docs: dict[str, set[str]],
    query_count: int,
    negative_example_size: int,
) -> None:
    with (
        jsonlines.open(ranking_file_path, mode="r") as reader,
        jsonlines.open(relevance_judgement_file_path, mode="w") as writer,
    ):
        for item in tqdm(reader, total=query_count):
            query_id = item.get("query_id")
            ranked_doc_ids = item.get("ranked_doc_ids")
            positive_doc_ids = relevant_docs.get(query_id, set())

            relevance_judgements = make_relevance_judgements(
                query_id, ranked_doc_ids, positive_doc_ids, negative_example_size
            )

            for relevance_judgement in relevance_judgements:
                writer.write(relevance_judgement._asdict())


def run(config: OmegaConf):
    logger.info("Creating relevance judgements...")

    relevance_judgement_file_path = Path(config.dataset.relevance_judgement_file_path)

    if not relevance_judgement_file_path.exists():
        relevant_docs = load_pickle(config.filter.filtered_relevant_docs_file_path)
        save_relevance_judgements(
            config.ranking.ranking_file_path,
            config.dataset.relevance_judgement_file_path,
            relevant_docs,
            config.dataset.query_count,
            config.dataset.negative_example_count,
        )
    else:
        logger.info("Passing relevance judgements as the file already exists")

    logger.info("Relevance judgements created")
