import os
import pickle
from pathlib import Path
from typing import Callable, List

import numpy as np
from smac.facade.smac_hpo_facade import SMAC4HPO as SMAC
from smac.scenario.scenario import Scenario

from afsfc.algorithms.config_space_composer import Mapper, build_config_space
from afsfc.measure import Measures
from afsfc.utils.string_utils import decode_parameter


class BinaryDataset:

    def __init__(self, path: str):
        self.base_path = path
        self.name = Path(path).name

    def load_dataset(self):
        return np.load(f"{self.base_path}/dataset.npy")

    def load_metafeatures(self):
        return np.load(f"{self.base_path}/meta.npy")


def extract_datasets(base_path: str):
    for root, dataset_folders, _ in os.walk(base_path):
        return dataset_folders


def _create_smac_directory(base_path: str, evaluator_name: str, config_name: str) -> str:
    dir_name: str = f"{base_path}/{evaluator_name}/smac/{config_name}"
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


class SmacPretrainer:
    def __init__(self, logger=None):
        self._smac = None
        self._logger = logger

    def fit(self, dataset: BinaryDataset,
            clustering_algs: List[str],
            feature_selection_algs: List[str],
            n_evaluations: int = 30,
            cutoff_time=20,
            evaluator: Callable = Measures.silhouette,
            experiments_dir: str = "../../experiments"):

        cs = build_config_space(clustering_ls=clustering_algs, feature_selection_ls=feature_selection_algs)

        config_name: str = "mixed"
        if len(clustering_algs) == 1 and len(feature_selection_algs) == 1:
            config_name: str = f"{feature_selection_algs[0]}_{clustering_algs[0]}"
        base_dir_name = _create_smac_directory(experiments_dir, evaluator.__name__, config_name)

        scenario_params: dict = {
            "run_obj": "quality",
            "runcount-limit": n_evaluations,
            "cutoff_time": cutoff_time,
            "cs": cs,
            "deterministic": "true",
            "output_dir": base_dir_name,
            "abort_on_first_run_crash": False,
        }
        scenario = Scenario(scenario_params)
        dataset_content = dataset.load_dataset()

        def fit_models(cfg: dict, data: np.ndarray):
            feature_selection_alg = Mapper.get_class(cfg["feature_selection_choice"])

            cfg_feature_selection: dict = {
                decode_parameter(k, feature_selection_alg.name): v for k, v in cfg.items()
                if decode_parameter(k, feature_selection_alg.name) is not None
            }

            feature_selection_model = feature_selection_alg.model(**cfg_feature_selection)
            selected_data: np.ndarray = feature_selection_model.fit_transform(data)

            clustering_alg = Mapper.get_class(cfg["clustering_choice"])
            cfg_clustering: dict = {
                decode_parameter(k, clustering_alg.name): v for k, v in cfg.items()
                if decode_parameter(k, clustering_alg.name) is not None
            }

            clustering_model = clustering_alg.model(**cfg_clustering)
            clustering_result = clustering_model.fit_transform(selected_data)

            return feature_selection_model, clustering_model, clustering_result

        def cfg_to_dict(cfg):
            cfg = {k: cfg[k] for k in cfg if cfg[k]}
            return {k: v for k, v in cfg.items() if v is not None}

        def evaluate_model(cfg):
            cfg_dict = cfg_to_dict(cfg)
            _, _, clustering_result = fit_models(cfg_dict, dataset_content)
            measure_value = evaluator(dataset_content, clustering_result)
            return measure_value

        optimal_config = None
        smac_params = {
            "scenario": scenario,
            "rng": np.random.RandomState(42),
            "tae_runner": evaluate_model,
            # "initial_configurations": None
        }
        smac = SMAC(**smac_params)
        self._smac = smac
        optimal_config = self._smac.optimize()

        feature_selection_model, clustering_model, clustering_result = \
            fit_models(cfg_to_dict(optimal_config), dataset_content)

        measure_value = evaluator(dataset_content, clustering_result)

        result = {
            "optimal_config": optimal_config,
            "smac": self._smac,
            "feature_selection_model": feature_selection_model,
            "clustering_model": clustering_model,
            "clustering_result": clustering_result,
            "measure_value": measure_value
        }
        _save_clustering_result(result, base_dir_name)

        return result


def _save_clustering_result(results: dict, base_path: str):
    with open(f"{base_path}/optimal_config.pkl", "wb") as f:
        pickle.dump(results["optimal_config"], f, pickle.HIGHEST_PROTOCOL)
    with open(f"{base_path}/clustering_result.npy", "wb") as f:
        np.save(f, results["clustering_result"])
    with open(f"{base_path}/clustering_result.txt", "w") as f:
        np.savetxt(f, results["measure_value"])


if __name__ == '__main__':
    meta_db_dir = "../../experiments/metadb"
    measures: [Callable] = [
        Measures.silhouette,
        Measures.davies_bouldin,
        Measures.calinski_harabasz
    ]
    clustering_algs: List[str] = [
        "DBSCAN", "KMeans", "MiniBatchKMeans", "AffinityPropagation",
        "MeanShift", "SpectralClustering", "AgglomerativeClustering",
        "OPTICS", "Birch", "GaussianMixture"
    ]
    feature_selection_algs: List[str] = [
        "Lasso", "LFSBSS", "MCFS", "WKMeans", "GenericSPEC", "FixedSPEC",
        "NormalizedCut", "NullModel"
    ]

    for dataset_folder in extract_datasets(meta_db_dir):
        dataset = BinaryDataset(f"{meta_db_dir}/{dataset_folder}")
        pretrainer = SmacPretrainer()
        for measure in measures:
            for feature_selection_alg in feature_selection_algs:
                for clustering_alg in clustering_algs:
                    try:
                        pretrainer.fit(
                            dataset,
                            clustering_algs=[clustering_alg],
                            feature_selection_algs=[feature_selection_alg],
                            n_evaluations=20,
                            cutoff_time=10,
                            evaluator=measure,
                            experiments_dir=f"{meta_db_dir}/{dataset_folder}")
                    except:
                        pass
