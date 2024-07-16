# PR-Rank

## Requirements

Please download the following resources:

- AOL-IA documents: Follow the instructions on [the ir-datasets website](https://ir-datasets.com/aol-ia.html)
- [FastText model](https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin) for filtering the AOL-IA dataset
- [DMOZ dataset](https://www.kaggle.com/datasets/lucmichalski/dmoz-bert-multiclass-web-classification-dataset/data) for training the document topic estimator
- [RankLib](https://sourceforge.net/p/lemur/wiki/RankLib%20Installation/) for training the Learning-to-Rank model

## Getting started

Install the required dependencies as follows:

```sh
conda env create -f env.yml
conda activate PR-Rank
pip install ./PR-Rank
```

## Running Experiments

To execute the series of experiments, run the following commands:

```sh
# Estimate qrel
python PR-Rank/aolia_qrel/main.py

# Extract features
python -m spacy download en_core_web_sm
python PR-Rank/features_extraction/main.py

# Divide dataset
python PR-Rank/dataset_division/main.py

# Train & evaluate PR-Rank (Parameter regression model)
python PR-Rank/parameter_regression/main.py
```

To modify experimental settings, edit the following configuration files:

- PR-Rank/aolia_qrel/config/config.yaml
- PR-Rank/features_extraction/config/config.yaml
- PR-Rank/dataset_division/config/config.yaml
- PR-Rank/parameter_regression/config/config.yaml

## Usage

PR-Rank involves two main experimental stages, each with its own configuration:

1. Dataset Division
2. PR-Rank Parameter Regression

### Changing Feature Sets

You can independently select feature sets for each experimental stage:

#### Dataset Division Feature Sets

In the dataset division configuration, modify the `feature_sets` parameter:

```yaml
# Use only query features for dataset division
feature_sets:
  - Q
```

#### PR-Rank Domain Feature Sets

In the PR-Rank configuration, modify the `domain_feature_sets` parameter:

```yaml
# Use all features sets for PR-Rank
domain_feature_sets:
  - Q
  - D
  - Q-D
```

Available options for both stages are Q (Query), D (Document), Q-D (Query-Document pair), or any combination.

### Experiment Naming Convention

Use descriptive names for each experimental stage to organize your runs effectively.

#### Dataset Division Experiment Name

In `PR-Rank/dataset_division/config/config.yaml`:

```yaml
# experiment_name: q
domains_dir_path: PR-Rank/dataset_division/experiment/q/data/domains
ltr_datasets_dir_path: PR-Rank/dataset_division/experiment/q/data/ltr_datasets
...
```

#### PR-Rank Experiment Name

In `PR-Rank/parameter_regression/config/config.yaml`:

```yaml
# experiment_name: all
domain_features_dir_path: PR-Rank/parameter_regression/experiment/all/data/domain_features
model_parameters_dir_path: PR-Rank/parameter_regression/experiment/all/data/model_parameters
...
```
