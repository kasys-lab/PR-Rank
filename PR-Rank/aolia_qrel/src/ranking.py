import json
import logging
import random
from collections import defaultdict
from pathlib import Path

import pyterrier as pt
from ir_datasets.datasets.base import Dataset
from omegaconf import OmegaConf
from tqdm import tqdm
from utils.aolia_utils import load_queries_store
from utils.file_utils import load_lines_as_set, load_pickle

logger = logging.getLogger(__name__)


def get_max_positive_example_count(
    query_have_relevant_docs: set[str], relevant_docs: dict[str, set[str]]
) -> int:
    positive_example_count2freq = defaultdict(int)

    for query_id in query_have_relevant_docs:
        positive_example_count = len(relevant_docs[query_id])
        positive_example_count2freq[positive_example_count] += 1

    max_count = max(positive_example_count2freq.keys(), default=0)

    return max_count


def save_ranking(
    query_limit: int,
    queries_store: dict[str, str],
    query_have_relevant_docs: set[str],
    model,
    ranking_file_path: Path,
) -> None:
    with open(ranking_file_path, "w") as f:
        query_count = 0

        shuffled_queries = list(query_have_relevant_docs)
        random.shuffle(shuffled_queries)

        for query_id in tqdm(shuffled_queries, total=query_limit):
            if query := queries_store.get(query_id, ""):
                res = model.search(query)

                if len(res) == 0:
                    continue

                line = {"query_id": query_id, "ranked_doc_ids": res["docno"].tolist()}
                json_line = json.dumps(line)
                f.write(json_line + "\n")

                query_count += 1

            if query_count >= query_limit:
                break


def run(dataset: Dataset, config: OmegaConf):
    logger.info("Ranking...")

    ranking_file_path = Path(config.ranking.ranking_file_path)

    if not ranking_file_path.exists():
        queries_store = load_queries_store(dataset, config.queries_store_file_path)

        filtered_query_ids = load_lines_as_set(
            config.filter.filtered_query_ids_file_path
        )
        relevant_docs = load_pickle(config.filter.filtered_relevant_docs_file_path)
        query_have_relevant_docs = set(relevant_docs.keys()) & filtered_query_ids

        # Create a BM25 model.
        index_dir_path = Path(config.ranking.index_dir_path)
        index_ref = pt.IndexRef.of(str(index_dir_path.absolute()))

        max_positive_example_count = get_max_positive_example_count(
            query_have_relevant_docs, relevant_docs
        )
        bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25") % (
            max_positive_example_count + config.dataset.negative_example_count
        )

        save_ranking(
            config.dataset.query_count,
            queries_store,
            query_have_relevant_docs,
            bm25,
            config.ranking.ranking_file_path,
        )
    else:
        logger.info("Passing ranking as the file already exists")

    logger.info("Ranking complete")
