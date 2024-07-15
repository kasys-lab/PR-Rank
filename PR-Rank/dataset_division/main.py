import logging

import hydra
from dataset_division.src import (
    calculate_q_level_features,
    cluster_queries,
    divide_dataset,
    eval_division,
    train_ltr_models
)
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: OmegaConf) -> None:
    calculate_q_level_features.run(config)
    cluster_queries.run(config)
    divide_dataset.run(config)
    train_ltr_models.run(config)
    eval_division.run(config)


if __name__ == "__main__":
    main()
