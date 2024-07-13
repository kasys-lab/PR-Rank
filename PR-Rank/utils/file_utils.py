import pickle
from pathlib import Path
from typing import Any, Iterable


def save_iter_to_lines(items: Iterable[Any], output_file_name: str | Path) -> None:
    with open(output_file_name, "w") as f:
        for item in items:
            f.write(f"{item}\n")


def load_lines_as_set(file_path: str | Path) -> set[str]:
    with open(file_path, "r") as f:
        return {line.strip() for line in f}


def save_pickle(obj: Any, output_file_name: str | Path) -> None:
    with open(output_file_name, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(file_path: str | Path) -> Any:
    with open(file_path, "rb") as f:
        return pickle.load(f)
