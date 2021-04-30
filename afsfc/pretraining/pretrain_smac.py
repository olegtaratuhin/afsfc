import os
import pickle
import traceback
from pathlib import Path
from typing import Callable, List

import numpy as np
# from smac.facade.smac_hpo_facade import SMAC4HPO as SMAC
# from smac.facade.smac_bohb_facade import SMAC4HPO as SMAC
# from smac.facade.smac_ac_facade import SMAC4AC as SMAC
from smac.facade.experimental.psmac_facade import PSMAC as SMAC
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


def _create_psmac_dirs(base_path: str, count: int) -> List[str]:
    created_dirs: List[str] = []
    for i in range(count):
        dir_name: str = f"{base_path}/psmac_{i}"
        os.makedirs(dir_name, exist_ok=True)
        created_dirs.append(dir_name)
    return created_dirs


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
            experiments_dir: str = "../../experiments",
            n_optimizers=2):

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
            "deterministic": "false",
            "output_dir": base_dir_name,
            "abort_on_first_run_crash": False,
            "shared_model": True,
            "input_psmac_dirs": _create_psmac_dirs(base_dir_name, n_optimizers)
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
            clustering_result = clustering_model.fit_predict(selected_data)

            return feature_selection_model, clustering_model, clustering_result

        def cfg_to_dict(cfg):
            cfg = {k: cfg[k] for k in cfg if cfg[k]}
            return {k: v for k, v in cfg.items() if v is not None}

        def evaluate_model(cfg):
            cfg_dict = cfg_to_dict(cfg)
            _, _, y_pred = fit_models(cfg_dict, dataset_content)
            if len(np.unique(y_pred)) < 2:
                return np.inf
            else:
                return evaluator(dataset_content, y_pred)

        optimal_config = None
        smac = SMAC(
            scenario=scenario,
            rng=np.random.RandomState(42),
            tae=evaluate_model,
            n_optimizers=n_optimizers
        )
        self._smac = smac
        optimal_config = self._smac.optimize()

        feature_selection_model, clustering_model, clustering_result = \
            fit_models(cfg_to_dict(optimal_config), dataset_content)

        if len(np.unique(clustering_result)) < 2:
            measure_value = np.inf
        else:
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
        np.savetxt(f, np.array([results["measure_value"]]))


if __name__ == '__main__':

    meta_db_dir = "../../experiments/metadb"
    measures: [Callable] = [Measures.silhouette]
    clustering_algs: List[str] = [
        "DBSCAN", "KMeans", "MiniBatchKMeans", "AffinityPropagation",
        "MeanShift", "SpectralClustering",  # "AgglomerativeClustering",
        "OPTICS", "Birch", "GaussianMixture"
    ]
    feature_selection_algs: List[str] = [
        "Lasso", "MCFS", "WKMeans", "GenericSPEC", "FixedSPEC",
        "NormalizedCut", "NullModel"
    ]
    meta_db = extract_datasets(meta_db_dir)
    finished_previously = ["banana", "bank-additional-full", "boston"]

    for dataset_folder in meta_db:
        dataset = BinaryDataset(f"{meta_db_dir}/{dataset_folder}")
        print(f"Got new dataset: {dataset.name}")
        if dataset.name in finished_previously:
            print(f"Skip {dataset.name}")
            continue
        for measure in measures:
            try:
                _ = SmacPretrainer().fit(
                    dataset,
                    clustering_algs=clustering_algs,
                    feature_selection_algs=feature_selection_algs,
                    n_evaluations=100,
                    cutoff_time=20,
                    evaluator=measure,
                    experiments_dir=f"{meta_db_dir}/{dataset_folder}",
                    n_optimizers=5
                )
            except:
                print(traceback.format_exc())
        break
