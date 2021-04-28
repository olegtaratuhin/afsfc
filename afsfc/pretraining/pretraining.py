import os
from datetime import datetime
from pathlib import Path
from typing import Callable, List

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


def _create_metalerning_directory(base_dir: str) -> str:
    dir_name: str = f"{base_dir}/metadb"
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


def _create_metadb_directory(dataset_name: str) -> str:
    dir_name: str = f"{base_dir}/{dataset_name}"
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


def _save_meta_description(dataset_dir: str, metadescription: np.ndarray):
    with open(f"{dataset_dir}/meta.npy", "wb") as f:
        np.save(f, np.array(metadescription))
    np.savetxt(f"{dataset_dir}/meta.txt", np.array(metadescription))


class Pretrainer:
    def __init__(self, logger=None):
        self._logger = logger

    def fit(self, dataset: Dataset,
            clustering_algs: List[str] = None,
            feature_selection_algs: List[str] = None,
            n_evaluations: int = 30,
            cutoff_time=20,
            optimizer: str = "smac",
            evaluator: Callable = Measures.silhouette):

        if clustering_algs is None:
            clustering_algs = ["KMeans", "DBSCAN"]
        if feature_selection_algs is None:
            feature_selection_algs = ["NullModel", "NormalizedCut"]

        n_clustering_cfgs = max([Mapper.get_class(alg).n_possible_cfgs for alg in clustering_algs])
        n_feature_selection_cfgs = max([Mapper.get_class(alg).n_possible_cfgs for alg in feature_selection_algs])
        n_evaluations = min(n_evaluations, n_clustering_cfgs * n_feature_selection_cfgs)
        cs = build_config_space(clustering_ls=clustering_algs, feature_selection_ls=feature_selection_algs)

        experiments_dir = "../../experiments"
        base_dir_name = _create_experiment_directory(experiments_dir)

        scenario_params: dict = {
            "run_obj": "quality",
            "runcount-limit": n_evaluations,
            "cutoff_time": cutoff_time,
            "cs": cs,
            "deterministic": "true",
            "output_dir": f"{base_dir_name}/smac",
            "abort_on_first_run_crash": False,
        }
        scenario = Scenario(scenario_params)


if __name__ == '__main__':
    cache_dir: str = "../../data/cache"
    config_dir: str = "../../data/real"

    outlier_removal = IsolationForestOutlierRemoval()
    feature_transformer = FeatureTransformer()
    metafeature_extractor = MetafeatureExtractor()
    measures: [Callable] = [
        Measures.silhouette,
        Measures.davies_bouldin
    ]

    base_dir = "../../experiments"
    metadb_dir = _create_metalerning_directory(base_dir)
    experiment_dir = _create_experiment_directory(base_dir)

    meta_table = pd.DataFrame()
    for dataset in extract_datasets(config_dir, cache_dir):
        try:
            dataset_name = Path(dataset.path).name
            dataset_name = dataset_name[:dataset_name.find(".")]
            dataset_db_dir = _create_metadb_directory(dataset_name)

            dataset: Dataset = feature_transformer.transform(dataset)
            dataset: Dataset = outlier_removal.remove_outliers(dataset)

            X = dataset.content.values
            metadescription = metafeature_extractor.extract_from_dataset(dataset)
            _save_meta_description(dataset_db_dir, metadescription)
        except:
            pass


