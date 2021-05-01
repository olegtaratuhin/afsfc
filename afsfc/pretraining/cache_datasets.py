import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable, List
from joblib import Parallel, delayed
import multiprocessing

import numpy as np
import pandas as pd
from smac.scenario.scenario import Scenario

from afsfc.algorithms.config_space_composer import Mapper, build_config_space
from afsfc.datasets import extract_datasets, Dataset
from afsfc.measure import Measures
from afsfc.metafeatures import MetafeatureExtractor
from afsfc.preprocessing import FeatureTransformer
from afsfc.preprocessing import IsolationForestOutlierRemoval


def _create_experiment_directory(base_dir: str) -> str:
    dir_name: str = f"{base_dir}/{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}"
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


def _create_metadb(base_dir: str, meta_db_name: str) -> str:
    dir_name: str = f"{base_dir}/{meta_db_name}"
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


def _create_metadb_directory(metadb_dir, dataset_name: str) -> str:
    dir_name: str = f"{metadb_dir}/{dataset_name}"
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


def _save_meta_description(dataset_dir: str, metadescription: np.ndarray):
    with open(f"{dataset_dir}/meta.npy", "wb") as f:
        np.save(f, np.array(metadescription))


def _save_preprocessed_dataset(dataset_db_dir: str, dataset: np.ndarray):
    with open(f"{dataset_db_dir}/dataset.npy", "wb") as f:
        np.save(f, np.array(dataset))


if __name__ == '__main__':
    cache_dir: str = "../../data/cache"
    config_dir: str = "../../data/synthetic"
    metadb_name: str = "metadb_synthetic"

    outlier_removal = IsolationForestOutlierRemoval()
    feature_transformer = FeatureTransformer()
    metafeature_extractor = MetafeatureExtractor()

    base_dir = "../../experiments"
    metadb_dir = _create_metadb(base_dir, metadb_name)

    def preproccess_dataset(dataset: Dataset):
        try:
            dataset_name = Path(dataset.path).name
            dataset_name = dataset_name[:dataset_name.find(".")]
            dataset_db_dir = _create_metadb_directory(metadb_dir, dataset_name)

            dataset: Dataset = feature_transformer.transform(dataset)
            dataset: Dataset = outlier_removal.remove_outliers(dataset)
            _save_preprocessed_dataset(dataset_db_dir, dataset.content.values)

            metadescription = metafeature_extractor.extract_from_dataset(dataset)
            _save_meta_description(dataset_db_dir, metadescription)
        except Exception as e:
            print(traceback.format_exc())

    n_cpus = multiprocessing.cpu_count()
    Parallel(prefer="processes", n_jobs=n_cpus)(delayed(preproccess_dataset)(dataset) for dataset in extract_datasets(config_dir, cache_dir))
