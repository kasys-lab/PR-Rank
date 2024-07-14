import logging

import hydra
import ir_datasets
from features_extraction.src import (
    extract_d_features,
    extract_q_features,
    extract_qd_features,
    preprocess_documents,
    preprocess_queries,
    train_topic_estimator,
)
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: OmegaConf) -> None:
    dataset = ir_datasets.load("aol-ia")

    train_topic_estimator.run(config)
    preprocess_documents.run(dataset, config)
    preprocess_queries.run(config)
    extract_d_features.run(config)
    extract_q_features.run(config)
    extract_qd_features.run(config)


if __name__ == "__main__":
    main()
