import pickle
from pathlib import Path
from typing import Any, Callable, Iterable

import jsonlines


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


def sort_jsonl(
    input_file: str, output_file: str, sort_key: Callable[[dict[str, Any]], Any]
) -> None:
    data: list[dict[str, Any]] = []
    with jsonlines.open(input_file) as reader:
        for item in reader:
            data.append(item)

    sorted_data = sorted(data, key=sort_key)

    with jsonlines.open(output_file, mode="w") as writer:
        writer.write_all(sorted_data)
