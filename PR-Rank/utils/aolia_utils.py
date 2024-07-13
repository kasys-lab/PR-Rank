import pickle
from pathlib import Path

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
