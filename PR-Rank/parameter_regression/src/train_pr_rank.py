import logging
from pathlib import Path

import optuna
import pandas as pd
from omegaconf import OmegaConf
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from utils.file_utils import save_pickle
from utils.ranklib_utils import eval_model, read_score, save_ltr_model

logger = logging.getLogger(__name__)


def rf_objective(
    trial,
    scaler,
    X_train,
    y_train,
    X_valid,
    datasets_dir_path,
    ltr_models_dir_path,
    ranklib_jar_path,
):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 14),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 14),
    }

    model = RandomForestRegressor(**params, random_state=42)
    model.fit(scaler.transform(X_train), y_train)
    score = eval_model_on_downstream_task(
        model,
        scaler,
        X_valid,
        datasets_dir_path,
        ltr_models_dir_path,
        ranklib_jar_path,
    )

    return score


META_MODELS = [
    ("RandomForest", RandomForestRegressor, rf_objective),
]


def load_dataset(feature_dir_path: Path, target_dir_path: Path, step: str):
    feature_file_path = Path(feature_dir_path) / f"{step}.csv"
    target_file_path = Path(target_dir_path) / f"{step}.csv"

    X = pd.read_csv(
        feature_file_path,
        header=None,
        index_col=0,
    )
    y = pd.read_csv(
        target_file_path,
        header=None,
        index_col=0,
    )
    return X.sort_index(), y.sort_index()


def eval_model_on_downstream_task(
    model, scaler, X, datasets_dir_path, ltr_models_dir_path, ranklib_jar_path
):
    # list of regressed parameters
    regressed_parameters = model.predict(scaler.transform(X))
    dataset_dir_paths = list(sorted(datasets_dir_path.glob("dataset_*")))

    scores = []
    for regressed_parameter, dataset_dir_path in zip(
        regressed_parameters,
        dataset_dir_paths,
    ):
        dataset_name = dataset_dir_path.name.split("_")[-1]
        ltr_model_dir_path = ltr_models_dir_path / f"dataset_{dataset_name}"
        ltr_model_dir_path.mkdir(parents=True, exist_ok=True)

        # save regressed parameter to model file
        test_file_path = dataset_dir_path / "test.txt"
        model_file_path = ltr_model_dir_path / "model.dat"
        idv_file_path = ltr_model_dir_path / "eval.txt"

        save_ltr_model(regressed_parameter, model_file_path)

        eval_model(ranklib_jar_path, test_file_path, model_file_path, idv_file_path)

        score = read_score(idv_file_path)[1]
        scores.append(score)

    average_score = sum(scores) / len(scores)
    return average_score


def run(config: OmegaConf):
    logger.info("Training PR-Rank...")

    feature_dir_path = config.domain_features_dir_path
    target_dir_path = config.model_parameters_dir_path
    models_dir_path = Path(config.pr_rank.models_dir_path)

    if not models_dir_path.exists():
        models_dir_path.mkdir(parents=True)

        X_train, y_train = load_dataset(feature_dir_path, target_dir_path, "train")
        X_valid, y_valid = load_dataset(feature_dir_path, target_dir_path, "valid")

        X_train_val = pd.concat([X_train, X_valid])
        y_train_val = pd.concat([y_train, y_valid])
        scaler = StandardScaler().fit(X_train_val)
        X_train_val_scaled = scaler.transform(X_train_val)
        save_pickle(scaler, models_dir_path / "scaler.pkl")

        for model_name, model, model_objective in META_MODELS:
            model_dir_path = models_dir_path / model_name
            model_dir_path.mkdir()

            # train model
            # test model on downstream task
            valid_datasets_dir_path = Path(config.ltr_datasets_dir_path) / "valid"
            models_with_regressed_parameters_dir_path = (
                model_dir_path / "models_with_regressed_parameters" / "valid"
            )
            models_with_regressed_parameters_dir_path.mkdir(parents=True)

            study = optuna.create_study(direction="maximize")
            study.optimize(
                lambda trial: model_objective(
                    trial,
                    scaler,
                    X_train,
                    y_train,
                    X_valid,
                    valid_datasets_dir_path,
                    models_with_regressed_parameters_dir_path,
                    config.ranklib_jar_path,
                ),
                n_trials=config.pr_rank.n_trials,
            )
            best_model = model(**study.best_params, random_state=42)
            best_model.fit(X_train_val_scaled, y_train_val)
            save_pickle(best_model, model_dir_path / "model.pkl")
    else:
        logger.info("model already exists")

    logger.info("Training PR-Rank completed.")
