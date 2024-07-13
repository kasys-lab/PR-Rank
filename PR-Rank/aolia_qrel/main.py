import logging
import os

import hydra
import ir_datasets
import pyterrier as pt
from aolia_qrel.src import (
    attach_qrels,
    detect_languages,
    filter_by_language,
    index,
    ranking,
)
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: OmegaConf) -> None:
    dataset = ir_datasets.load("aol-ia")

    # filter dataset
    detect_languages.run(dataset, config)
    filter_by_language.run(dataset, config)

    # Initialize PyTerrier
    os.environ["JAVA_HOME"] = config.java_home_dir_path
    if not pt.started():
        pt.init()

    # ranking
    index.run(dataset, config)
    ranking.run(dataset, config)

    # attach qrels
    attach_qrels.run(config)


if __name__ == "__main__":
    main()
