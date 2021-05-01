import json
from typing import List

from joblib import Parallel, delayed
from numpy.random import default_rng

from afsfc.datasets import Dataset, generate_classification, \
    generate_moons, generate_circles, generate_blobs


def save_datasets(base_config_path: str, base_cache_path: str, datasets: List[Dataset], prefix: str):
    for i, dataset in enumerate(datasets):
        dataset.path = f"{prefix}_{i:03d}.csv"
        dataset.content.to_csv(f"{base_cache_path}/{dataset.path}")
        dataset.content = None

        with open(f"{base_config_path}/{prefix}_{i}.json", 'w', encoding='utf-8') as f:
            json.dump(dataset.to_dict(), f)


if __name__ == '__main__':
    base_config_path: str = "../../data/synthetic"
    base_cache_path: str = "../../data/cache"
    random_state = 42
    rng = default_rng(random_state)

    def _generate_blobs():
        print("Generating blobs")
        blobs_count = 200
        blobs_datasets: List[Dataset] = []
        for _ in range(blobs_count):
            params = {
                "n_samples": rng.integers(low=100, high=2000),
                "n_features": rng.integers(low=20, high=250),
                "n_clusters": rng.integers(low=2, high=20),
                "random_state": random_state
            }
            dataset: Dataset = generate_blobs(**params)
            blobs_datasets.append(dataset)
        save_datasets(base_config_path, base_cache_path, datasets=blobs_datasets, prefix="blobs")
        del blobs_datasets
        print("Blobs generation done")

    def _generate_circles():
        print("Generating circles")
        circles_count = 50
        circles_datasets: List[Dataset] = []
        for _ in range(circles_count):
            params = {
                "n_samples": tuple(rng.integers(low=200, high=2000, size=2)),
                "random_state": random_state,
                "noise": rng.random()
            }
            dataset: Dataset = generate_circles(**params)
            circles_datasets.append(dataset)
        save_datasets(base_config_path, base_cache_path, datasets=circles_datasets, prefix="circles")
        del circles_datasets
        print("Circles generation done")

    def _generate_classification():
        print("Generating classification")
        classification_count = 400
        classification_datasets: List[Dataset] = []
        for _ in range(classification_count):
            params = {
                "n_samples": rng.integers(low=250, high=2000),
                "n_informative": rng.integers(low=20, high=100),
                "n_redundant": rng.integers(low=20, high=100),
                "n_clusters": rng.integers(low=2, high=50),
                "n_repeated": rng.integers(low=10, high=50),
                "random_state": random_state
            }
            dataset: Dataset = generate_classification(**params)
            classification_datasets.append(dataset)
        save_datasets(base_config_path, base_cache_path, datasets=classification_datasets, prefix="clf")
        del classification_datasets
        print("Classification generation done")

    def _generate_moons():
        print("Generating moons")
        moons_count = 200
        moons_datasets: List[Dataset] = []
        for _ in range(moons_count):
            params = {
                "n_samples": tuple(rng.integers(low=100, high=2000, size=2)),
                "random_state": random_state,
                "noise": rng.random()
            }
            dataset: Dataset = generate_moons(**params)
            moons_datasets.append(dataset)
        save_datasets(base_config_path, base_cache_path, datasets=moons_datasets, prefix="moons")
        del moons_datasets
        print("Moons generation done")

    jobs = [
        _generate_blobs,
        _generate_circles,
        _generate_classification,
        _generate_moons
    ]

    Parallel(n_jobs=4)(delayed(job)() for job in jobs)
