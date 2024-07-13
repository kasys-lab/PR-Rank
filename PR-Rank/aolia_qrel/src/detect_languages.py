import logging
from pathlib import Path
from typing import Dict, Generator

import fasttext
import jsonlines
from ir_datasets.datasets.base import Dataset
from omegaconf import OmegaConf
from tqdm import tqdm

logger = logging.getLogger(__name__)


def detect_language(model: fasttext.FastText, text: str) -> str:
    if text == "":
        return ""
    else:
        predicted_label = model.predict(text)
        language = predicted_label[0][0]
        return language


def detect_document_languages(
    dataset: Dataset, model: fasttext.FastText
) -> Generator[Dict[str, str], None, None]:
    for doc in dataset.docs_iter():
        yield {
            "doc_id": doc.doc_id,
            "title_language": detect_language(model, doc.title),
            "text_language": detect_language(model, doc.text),
        }


def detect_query_languages(
    dataset: Dataset, model: fasttext.FastText
) -> Generator[Dict[str, str], None, None]:
    for query in dataset.queries_iter():
        yield {
            "query_id": query.query_id,
            "text_language": detect_language(model, query.text),
        }


def run(dataset, config: OmegaConf):
    logger.info("Detecting languages")

    model = fasttext.load_model(config.filter.fasttext_model_file_path)

    document_languages_file_path = Path(config.filter.query_languages_file_path)
    if not document_languages_file_path.exists():
        with jsonlines.open(document_languages_file_path, "w") as writer:
            for doc in tqdm(
                detect_document_languages(dataset, model), total=dataset.docs_count()
            ):
                writer.write(doc)
    else:
        logger.info("Passing document language detection as the file already exists")

    query_languages_file_path = Path(config.filter.query_languages_file_path)
    if not query_languages_file_path.exists():
        with jsonlines.open(query_languages_file_path, "w") as writer:
            for query in tqdm(
                detect_query_languages(dataset, model), total=dataset.queries_count()
            ):
                writer.write(query)
    else:
        logger.info("Passing query language detection as the file already exists")

    logger.info("Languages detected")
