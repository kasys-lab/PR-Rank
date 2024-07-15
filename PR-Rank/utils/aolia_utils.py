import pickle
from pathlib import Path

import jsonlines
from ir_datasets.datasets.base import Dataset
from tqdm import tqdm


def load_queries_store(
    dataset: Dataset, queries_store_pickle_path: str | Path
) -> dict[str, str]:
    queries_store_pickle_path = Path(queries_store_pickle_path)

    if queries_store_pickle_path.exists():
        print("loading queries store...")
        with open(queries_store_pickle_path, "rb") as f:
            queries_store = pickle.load(f)
    else:
        queries_store = {}
        print("creating queries store...")

        for query in tqdm(dataset.queries_iter(), total=dataset.queries_count()):
            queries_store[query.query_id] = query.text

        with open(queries_store_pickle_path, "wb") as f:
            pickle.dump(queries_store, f)

    return queries_store


def get_dataset_for_experiment(relevance_judgement_file_path: Path) -> tuple[set, set]:
    doc_ids_expt = set()
    query_ids_expt = set()

    with jsonlines.open(relevance_judgement_file_path, "r") as reader:
        for item in tqdm(reader):
            doc_ids_expt.add(item["doc_id"])
            query_ids_expt.add(item["query_id"])

    return doc_ids_expt, query_ids_expt


FEATURES_INDEX = {
    "Q": [i for i in range(0, 6+1)],
    "D": [i for i in range(7, 28+1)],
    "Q-D": [i for i in range(29, 58 + 1)],
    "LtR": [7, 8, 9, 10, 26, 27, 28] + [i for i in range(29, 58 + 1)],
}
