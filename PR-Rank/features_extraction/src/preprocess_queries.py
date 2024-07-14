import json
import logging
from dataclasses import dataclass
from pathlib import Path

import spacy
from omegaconf import OmegaConf
from tqdm import tqdm
from utils.aolia_utils import get_dataset_for_experiment, load_queries_store
from utils.text_utils import shorten_text

logger = logging.getLogger(__name__)


@dataclass
class QueryDetail:
    query_id: str
    tokens: list[tuple[str, str]]

    def to_json(self):
        return json.dumps(self.__dict__)


def preprocess(
    queries_store,
    query_ids: set[str],
    query_details_file_path: Path,
    text_length_limit: int,
) -> None:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    logger.info("Start preprocessing...")

    with open(query_details_file_path, "w") as f:
        for query_id in tqdm(query_ids):
            try:
                query = queries_store.get(query_id)
            except KeyError:
                logger.info(f"query_id {query_id} not found")
                continue

            try:
                parsed_query = nlp(shorten_text(query, text_length_limit))

                query_tokens = [token.lemma_ for token in parsed_query]

                query_detail = QueryDetail(
                    query_id=query_id,
                    tokens=query_tokens,
                )

                f.write(query_detail.to_json() + "\n")

            except ValueError:
                logger.info(f"query_id {query_id} has {len(query)} chars, skipped")
                continue


def run(config: OmegaConf):

    query_details_file_path = Path(config.data.query_details_file_path)
    if not query_details_file_path.exists():
        queries_store = load_queries_store(config.data.queries_store_file_path)

        _doc_ids_expt, query_ids_expt = get_dataset_for_experiment(
            config.data.relevance_judgement_file_path
        )
        logger.info(f"preprocessing {len(query_ids_expt)} queries...")


        preprocess(
            queries_store,
            query_ids_expt,
            query_details_file_path,
            logger,
        )
    else:
        logger.info(f"{query_details_file_path} already exists, skipping preprocessing")
