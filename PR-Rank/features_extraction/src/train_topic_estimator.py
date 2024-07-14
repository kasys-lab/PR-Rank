import logging
from pathlib import Path

import pandas as pd
import spacy
from omegaconf import OmegaConf
from scipy.stats import loguniform, uniform
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from spacy.tokens import Doc
from utils.file_utils import load_pickle, save_pickle

logger = logging.getLogger(__name__)


class LemmaTokenizer:
    def __init__(self):
        # Load the model once during the initialization
        self.spacynlp = spacy.load("en_core_web_sm")

    def __call__(self, doc: str) -> list[str]:
        nlpdoc: Doc = self.spacynlp(doc)
        # Filter out stop words and punctuation if needed, and ensure tokens are alphanumeric
        lemmas = [
            token.lemma_.lower()
            for token in nlpdoc
            if not token.is_stop and not token.is_punct and token.lemma_.isalnum()
        ]
        return lemmas


def filter_df(
    df: pd.DataFrame,
    language: str,
    lang_confidence_threshold: float,
) -> pd.DataFrame:
    # Filter rows based on language and language confidence
    filtered_df = df[
        (df["language"] == language)
        & (df["language_confidence"] >= lang_confidence_threshold)
    ].copy()

    # Combine title and description into a single text column
    filtered_df.loc[:, "text"] = (
        filtered_df["title"].astype(str) + " " + filtered_df["description"].astype(str)
    )

    # Select the columns of interest
    df_learning_data = filtered_df[["topic_main", "text"]].copy()

    return df_learning_data.drop_duplicates()


def vectorize_and_split_data(
    df, test_size: float, random_state: int, n_samples=None, vectorizer_file_path=None
):
    if n_samples is not None and n_samples < len(df):
        df = shuffle(df, random_state=random_state).reset_index(drop=True)
        df = df.iloc[:n_samples]

    X = df["text"]
    y = df["topic_main"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if vectorizer_file_path and Path(vectorizer_file_path).exists():
        logger.info("Loading existing vectorizer...")
        pipe = load_pickle(vectorizer_file_path)
    else:
        logger.info("Creating new vectorizer...")
        pipe = Pipeline(
            [
                ("vect", CountVectorizer(tokenizer=LemmaTokenizer())),
                ("tfidf", TfidfTransformer()),
            ]
        )
        logger.info("Fitting the vectorizer...")
        pipe.fit(X_train)
        save_pickle(pipe, vectorizer_file_path)

    logger.info("Transforming the train data...")
    X_train_tfidf = pipe.transform(X_train)

    logger.info("Transforming the test data...")
    X_test_tfidf = pipe.transform(X_test)

    return X_train_tfidf, X_test_tfidf, y_train, y_test


def report_model_performance(clf, X_test, y_test, report_file_path):
    y_pred = clf.predict(X_test)

    confusion = f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n"
    class_report = f"Classification Report:\n{classification_report(y_test, y_pred)}\n"

    with open(report_file_path, "w") as file:
        file.write(confusion)
        file.write(class_report)


def run(config: OmegaConf):
    logger.info("Training topic estimator...")

    model_file_path = Path(config.topic_estimator.model_file_path)
    vectorizer_file_path = Path(config.topic_estimator.vectorizer_file_path)

    if not model_file_path.exists():
        df = pd.read_csv(config.topic_estimator.dmoz_file_path, sep="\t")
        df = filter_df(
            df,
            config.topic_estimator.language,
            config.topic_estimator.lang_confidence_threshold,
        )

        # feature extraction
        X_train, X_test, y_train, y_test = vectorize_and_split_data(
            df,
            test_size=config.topic_estimator.test_size,
            random_state=42,
            n_samples=config.topic_estimator.sample_count,
            vectorizer_file_path=vectorizer_file_path,
        )

        # train topic estimator
        clf = LogisticRegression(random_state=42, max_iter=1000)

        param_distributions = {
            "C": loguniform(1e-3, 1e3),
            "penalty": ["l1", "l2", "elasticnet"],
            "solver": ["saga"],
            "l1_ratio": uniform(0, 1),
        }

        random_search = RandomizedSearchCV(
            clf,
            param_distributions=param_distributions,
            n_iter=100,  # 試行回数
            cv=5,
            scoring="f1_macro",
            n_jobs=-1,
            verbose=2,
            random_state=42,
        )

        logger.info("Starting hyperparameter tuning...")
        random_search.fit(X_train, y_train)

        logger.info(f"Best parameters: {random_search.best_params_}")
        logger.info(f"Best cross-validation score: {random_search.best_score_:.4f}")

        best_model = random_search.best_estimator_
        save_pickle(best_model, model_file_path)

        report_model_performance(
            best_model, X_test, y_test, config.topic_estimator.eval_file_path
        )
    else:
        logger.info(f"Model already exists in {model_file_path}")

    logger.info("Training complete")
