import logging

import hydra
from omegaconf import OmegaConf
from parameter_regression.src import (
    extract_domain_features,
    extract_model_parameters,
    train_global_model,
    train_pr_rank,
)

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: OmegaConf) -> None:
    extract_domain_features.run(config)
    extract_model_parameters.run(config)
    train_global_model.run(config)
    train_pr_rank.run(config)

if __name__ == "__main__":
    main()
