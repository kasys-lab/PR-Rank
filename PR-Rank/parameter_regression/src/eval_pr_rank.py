import logging
from collections import defaultdict
from pathlib import Path

import ir_measures
from ir_measures import MAP, nDCG
from omegaconf import OmegaConf
from parameter_regression.src.train_pr_rank import (
    eval_model_on_downstream_task,
    load_dataset,
)
from utils.file_utils import load_pickle
from utils.ranklib_utils import rank_by_model, read_score

logger = logging.getLogger(__name__)


# Implement ERR by myself because ir-measures.ERR is bugged
class ERR:
    def __init__(self, cutoff=10):
        self.cutoff = cutoff

    def iter_calc(self, qrels, run):
        # Convert qrels to a dict for faster access
        qrel_dict = {(qrel.query_id, qrel.doc_id): qrel.relevance for qrel in qrels}

        run_by_query = defaultdict(list)
        for scored_doc in run:
            run_by_query[scored_doc.query_id].append(scored_doc)

        # Sort the documents for each query by score in descending order
        for query_id in run_by_query:
            run_by_query[query_id].sort(key=lambda x: x.score, reverse=True)

        # Calculate ERR for each query
        for query_id, docs in run_by_query.items():
            err = 0.0
            p = 1.0
            for rank, doc in enumerate(docs[: self.cutoff], start=1):
                relevance = qrel_dict.get((query_id, doc.doc_id), 0)
                # Convert relevance grade to probability of relevance
                max_relevance = max(
                    qrel.relevance for qrel in qrels if qrel.query_id == query_id
                )
                r = (2**relevance - 1) / 2**max_relevance
                err += p * r / rank
                p *= 1 - r
            yield (query_id, f"ERR@{self.cutoff}", err)


def qrel_line2instance(line: str) -> ir_measures.Qrel:
    try:
        contents, comment = line.split("#")[0], line.split("#")[1]
    except IndexError:
        contents, comment = line, "doc_id=_"
    parts = contents.split()
    query_id = parts[1].split(":")[1]
    doc_id = comment.split("=")[1].strip()
    relevance = int(parts[0])
    return ir_measures.Qrel(query_id, doc_id, relevance)


def scored_doc_line2instance(line: str, doc_id: str) -> ir_measures.ScoredDoc:
    split_line = line.strip().split()
    query_id = split_line[0]
    score = float(split_line[2])
    return ir_measures.ScoredDoc(query_id, doc_id, score)


def load_qrels_run(test_file_path, score_file_path):
    qrels = []
    run = []

    with (
        test_file_path.open(mode="r") as qrels_file,
        score_file_path.open(mode="r") as scored_docs_file,
    ):
        for qrel_line, scored_doc_line in zip(qrels_file, scored_docs_file):
            qrel_instance = qrel_line2instance(qrel_line)
            scored_doc_instance = scored_doc_line2instance(
                scored_doc_line, qrel_instance.doc_id
            )
            assert qrel_instance.query_id == scored_doc_instance.query_id
            assert qrel_instance.doc_id == scored_doc_instance.doc_id
            qrels.append(qrel_instance)
            run.append(scored_doc_instance)

    return qrels, run


def write_result(idv_file_path: Path, result) -> None:
    total = 0
    cnt = 0
    with idv_file_path.open(mode="w") as f:
        for query_id, measure, value in result:
            total += value
            cnt += 1
            f.write(f"{query_id} {measure} {value}\n")
        f.write(f"all {measure} {total / cnt}\n")


def eval_models_by_the_same_dataset(
    dataset_info: tuple[str, Path],
    models_info: list[tuple[str, Path, Path]],
    metrics: dict,
    ranklib_jar_path,
    override: bool = False,
) -> dict[str, dict[str, float]]:
    # scores = {model: {metric: val}}
    scores: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    dataset_name, test_file_path = dataset_info

    if not test_file_path.exists():
        raise FileNotFoundError(f"test file not found: {test_file_path}")

    for model_name, model_file_path, idv_dir_path in models_info:
        score_file_path = idv_dir_path / f"score_in_{dataset_name}.txt"

        # memoization
        if not score_file_path.exists() or override:
            rank_by_model(
                ranklib_jar_path,
                test_file_path,
                model_file_path,
                score_file_path,
            )

        for metric_name, metric in metrics.items():
            idv_file_path = idv_dir_path / f"eval_in_{dataset_name}.{metric_name}.txt"

            # memoization
            if not idv_file_path.exists() or override:
                qrels, run = load_qrels_run(test_file_path, score_file_path)
                result = metric.iter_calc(qrels, run)
                write_result(idv_file_path, result)

            try:
                _, score = read_score(idv_file_path)
            except UnboundLocalError:
                print(f"{idv_file_path} may be empty")
                exit()

            scores[model_name][metric_name] = score

    return scores


def eval_models_by_variety_of_domains(
    datasets_info: list[tuple[str, Path]],
    models_info: list[tuple[int, str, Path, Path]],
    ideal_models_dir_path: Path,
    pr_rank_output_dir_path: Path,
    metrics: dict,
    ranklib_jar_path,
    override: bool = False,
    logger=None,
) -> dict[str, dict[str, dict[str, float]]]:
    # result = {dataset_name: scores}
    result = {}

    for dataset_info in datasets_info:
        dataset_name = dataset_info[0]

        # Ideal model
        ideal_model_dir_path = ideal_models_dir_path / f"dataset_{dataset_name}"
        ideal_model_file_path = ideal_model_dir_path / "model.dat"
        ideal_model_info = ("Ideal", ideal_model_file_path, ideal_model_dir_path)
        # PR-Rank
        pr_idv_dir_path = pr_rank_output_dir_path / f"dataset_{dataset_name}"
        pr_rank_file_path = pr_idv_dir_path / "model.dat"
        pr_rank_info = ("PR-Rank", pr_rank_file_path, pr_idv_dir_path)

        try:
            scores = eval_models_by_the_same_dataset(
                dataset_info,
                models_info + [ideal_model_info, pr_rank_info],
                metrics,
                ranklib_jar_path,
                override,
            )
            result[dataset_name] = scores
        except FileNotFoundError:
            print(f"dataset {dataset_name} not found")
            if logger is not None:
                logger.info(f"dataset {dataset_name} not found")
            continue

    return result


def convert_to_markdown_tables(data, output_file_path: Path):
    average_metrics = {}
    all_models = set()
    metrics = list(data[next(iter(data))][next(iter(data[next(iter(data))]))].keys())

    # 平均値の計算とモデル名の収集
    for testdata, models in data.items():
        for model, model_metrics in models.items():
            all_models.add(model)
            if model not in average_metrics:
                average_metrics[model] = {
                    metric: [0, 0] for metric in metrics
                }  # [sum, count]
            for metric, value in model_metrics.items():
                average_metrics[model][metric][0] += value
                average_metrics[model][metric][1] += 1

    markdown_tables = "# Results\n\n"

    markdown_tables += create_table("Average Metrics", average_metrics, metrics)

    sorted_testdata = sorted(data.keys())
    for testdata in sorted_testdata:
        markdown_tables += create_table(testdata, data[testdata], metrics)

    with Path(output_file_path).open(mode="w") as f:
        f.write(markdown_tables)


def create_table(title, data, metrics):
    table = f"## {title}\n\n| Model |"
    for metric in metrics:
        table += f" {metric} |"
    table += "\n|" + "------|" * (len(metrics) + 1) + "\n"

    sorted_models = sorted(data.keys())
    for model in sorted_models:
        table += f"| {model} |"
        model_metrics = data[model]
        for metric in metrics:
            if title == "Average Metrics":
                metric_sum, metric_count = model_metrics[metric]
                value = metric_sum / metric_count if metric_count > 0 else 0
            else:
                value = model_metrics[metric]
            table += f" {value:.4f} |"
        table += "\n"
    table += "\n"
    return table


def run(config: OmegaConf):
    # === construct ltr models ===
    logging.info("Constructing LtR models by PR-Rank...")
    parameter_regressor = load_pickle(config.eval.pr_rank_file_path)
    scaler = load_pickle(config.pr_rank.scaler_file_path)
    X_test, _y_test = load_dataset(
        config.domain_features_dir_path, config.model_parameters_dir_path, "test"
    )
    _score = eval_model_on_downstream_task(
        parameter_regressor,
        scaler,
        X_test,
        Path(config.eval.test_datasets_dir_path),
        Path(config.eval.pr_rank_output_dir_path),
        config.ranklib_jar_path,
    )

    # === eval models ===
    logging.info("Eval LtR models...")
    metrics = {
        "nDCG@10": nDCG @ 10,
        "ERR@10": ERR(cutoff=10),
        "MAP": MAP(rel=config.eval.test_dataset_highest_rel),
    }

    # (dataset_nam, dataset_file_path)
    datasets_info: list[tuple[str, Path]] = []

    datasets_dir_path = Path(config.eval.test_datasets_dir_path)
    for dataset_dir_path in datasets_dir_path.glob("dataset_*"):
        dataset_name = dataset_dir_path.name.split("_")[-1]
        dataset_file_path = dataset_dir_path / "test.txt"
        datasets_info.append((dataset_name, dataset_file_path))

    # (model_nam, model_path, idv_dir_path)
    models_info: list[tuple[str, Path, Path]] = []

    ## Add global model
    large_model_dir_path = Path(config.global_model.global_model_dir_path)
    large_model_idv_dir_path = large_model_dir_path / "eval_by_other_domain"
    large_model_idv_dir_path.mkdir(parents=True, exist_ok=True)
    models_info.append(
        (
            "Global",
            large_model_dir_path / "model.dat",
            large_model_idv_dir_path,
        )
    )

    ## Add ideal models
    ideal_models_dir_path = Path(config.eval.ideal_models_dir_path)

    ## Add PR-Rank model
    pr_rank_dir_path = Path(config.eval.pr_rank_output_dir_path)

    # eval all models by variety of domains
    result = eval_models_by_variety_of_domains(
        datasets_info,
        models_info,
        ideal_models_dir_path,
        pr_rank_dir_path,
        metrics,
        config.ranklib_jar_path,
        override=False,
    )

    convert_to_markdown_tables(result, config.eval.result_file_path)
