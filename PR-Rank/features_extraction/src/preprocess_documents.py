import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from string import punctuation, whitespace

import ir_datasets
import publicsuffix2
import spacy
from features_extraction.src.collection import Collection
from ir_datasets.datasets.base import Dataset
from omegaconf import OmegaConf
from tqdm import tqdm
from utils.aolia_utils import get_dataset_for_experiment
from utils.text_utils import shorten_text

logger = logging.getLogger(__name__)

PUNCTUATION = set(list(punctuation))
WHITESPACE = set(list(whitespace))
INVALID_TOKENS = PUNCTUATION | WHITESPACE


@dataclass
class DocumentDetail:
    doc_id: str
    text_tokens: list[tuple[str, str]]
    title_tokens: list[tuple[str, str]]
    sentence_count: int
    second_level_domain: str

    def to_json(self):
        return json.dumps(self.__dict__)


def extract_lemma_and_pos(doc: spacy.tokens.Doc) -> list[tuple[str, str]]:
    lemma_pos_pairs = [(token.lemma_, token.pos_) for token in doc]
    return lemma_pos_pairs


def preprocess(
    docs_store: ir_datasets.indices.lz4_pickle.PickleLz4FullStore,
    doc_ids: set[str],
    document_details_file_path: Path,
    collection_file_path: Path,
    sld_counter_file_path: Path,
    text_length_limit: int,
) -> None:
    collection = Collection()
    sld_counter = defaultdict(int)
    psl_file = publicsuffix2.fetch()
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    logger.info("Start preprocessing...")

    with open(document_details_file_path, "w") as f:
        for doc_id in tqdm(doc_ids):
            try:
                doc = docs_store.get(doc_id)
            except KeyError:
                logger.info(f"doc_id {doc_id} not found")
                continue

            try:
                # save document details
                parsed_text = nlp(shorten_text(doc.text, text_length_limit))
                parsed_title = nlp(shorten_text(doc.title, text_length_limit))

                text_tokens = extract_lemma_and_pos(parsed_text)
                title_tokens = extract_lemma_and_pos(parsed_title)

                sentence_count = sum([1 for _ in parsed_text.sents])
                domain_name = doc.url.split("/")[2]
                sld = publicsuffix2.get_sld(domain_name, psl_file)

                docment_detail = DocumentDetail(
                    doc_id=doc_id,
                    text_tokens=text_tokens,
                    title_tokens=title_tokens,
                    second_level_domain=sld,
                    sentence_count=sentence_count,
                )

                f.write(docment_detail.to_json() + "\n")

                # update Collection, DomainCounter
                token_freq_in_text = Counter(
                    [token.lemma_.lower() for token in parsed_text]
                )
                token_freq_in_title = Counter(
                    [token.lemma_.lower() for token in parsed_title]
                )
                collection.add(token_freq_in_text)
                collection.add(token_freq_in_title)
                sld_counter[sld] += 1

            except ValueError:
                logger.info(f"doc_id {doc_id} has {len(doc.text)} chars, skipped")
                continue

    with open(collection_file_path, "w") as f:
        collection.dump(f)

    with open(sld_counter_file_path, "w") as f:
        json.dump(dict(sld_counter), f)


def run(dataset: Dataset, config: OmegaConf):
    logger.info("Preprocessing documents...")

    document_details_file_path = Path(config.preprocess.document_details_file_path)

    if not document_details_file_path.exists():
        docs_store = dataset.docs_store()
        doc_ids_expt, _query_ids_expt = get_dataset_for_experiment(
            config.data.relevance_judgement_file_path
        )
        doc_ids_expt = list(sorted(list(doc_ids_expt), key=lambda x: int(x, 16)))
        logger.info(f"preprocess {len(doc_ids_expt)} documents...")
        preprocess(
            docs_store,
            doc_ids_expt,
            document_details_file_path,
            config.preprocess.collection_file_path,
            config.preprocess.sld_counter_file_path,
            config.preprocess.text_length_limit,
        )
    else:
        logger.info("Document details already exist.")

    logger.info("Preprocessing finished!")
