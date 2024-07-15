# PR-Rank

## Requirements

Please download the following resources:

- AOL-IA documents: Follow the instructions on [the ir-datasets website](https://ir-datasets.com/aol-ia.html)
- [FastText model](https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin) for filtering the AOL-IA dataset
- [DMOZ dataset](https://www.kaggle.com/datasets/lucmichalski/dmoz-bert-multiclass-web-classification-dataset/data) for training the document topic estimator
- [RankLib](https://sourceforge.net/p/lemur/wiki/RankLib%20Installation/) for training the Learning-to-Rank model

## Setup

Install the required dependencies as follows:

```sh
conda env create -n PR-Rank -f env.yml
conda activate PR-Rank
pip install ./PR-Rank
```

## Running Experiments

To execute the series of experiments, run the following code. If you wish to modify the experimental settings, please edit these configuration files:

- PR-Rank/aolia_qrel/config/config.yaml
- PR-Rank/features_extraction/config/config.yaml
- PR-Rank/dataset_division/config/config.yaml
- PR-Rank/parameter_regression/config/config.yaml

Execute the following commands:

```sh
# Estimate qrel
python PR-Rank/aolia_qrel/main.py

# Extract features
python -m spacy download en_core_web_sm
python PR-Rank/features_extraction/main.py

# Divide dataset
python PR-Rank/dataset_division/main.py

# Train & evaluate PR-Rank
python PR-Rank/parameter_regression/main.py
```
