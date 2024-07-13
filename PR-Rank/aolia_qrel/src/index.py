import logging
from pathlib import Path
from typing import Generator

import pyterrier as pt
from ir_datasets.datasets.base import Dataset
from omegaconf import OmegaConf
from tqdm import tqdm
from utils.file_utils import load_lines_as_set

logger = logging.getLogger(__name__)


def generate_filtered_docs(
    dataset: Dataset, filtered_doc_ids: set[str]
) -> Generator[dict[str, str], None, None]:
    docs_store = dataset.docs_store()
    for doc_id in tqdm(filtered_doc_ids, total=len(filtered_doc_ids)):
        doc = docs_store.get(doc_id)
        if doc is not None:
            yield {"docno": doc.doc_id, "title": doc.title, "text": doc.text}


def run(dataset: Dataset, config: OmegaConf):
    logger.info("Indexing filtered documents...")

    index_dir_path = Path(config.ranking.index_dir_path)

    if not index_dir_path.exists():
        filtered_doc_ids = load_lines_as_set(config.filter.filtered_doc_ids_file_path)
        indexer = pt.IterDictIndexer(str(index_dir_path.absolute()))
        index_dir_path.mkdir(parents=True)
        _index_ref = indexer.index(
            generate_filtered_docs(dataset, filtered_doc_ids), fields=["title", "text"]
        )
    else:
        logger.info(f"Index already exists in {index_dir_path}")

    logger.info("Indexing complete")
