from typing import Tuple, Union

from sklearn import datasets
import pandas as pd
from afsfc.datasets import Dataset


def create_dataset_from_df(df: pd.DataFrame) -> Dataset:
    dataset: Dataset = Dataset(
        categorical=[],
        ignore=[],
        numerical=list(map(str, df.columns)),
        ordinal={},
        classification=None,
        target=None,
        url=None,
        path="",
    )
    dataset.content = df
    return dataset


def generate_blobs(n_samples: int = 10000, n_features: int = 100,
                   n_clusters: int = 5, random_state: int = 42) -> Dataset:
    df: pd.DataFrame = pd.DataFrame(datasets.make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        random_state=random_state
    )[0])
    return create_dataset_from_df(df)


def generate_classification(n_samples: int = 10000, n_informative: int = 20,
                            n_redundant: int = 20, random_state: int = 42,
                            n_clusters: int = 5, n_repeated: int = 20) -> Dataset:
    df: pd.DataFrame = pd.DataFrame(datasets.make_classification(
        n_samples=n_samples,
        n_features=n_repeated + n_redundant + n_informative,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_clusters,
        n_repeated=n_repeated,
        random_state=random_state
    )[0])
    return create_dataset_from_df(df)


def generate_moons(n_samples: Union[int, Tuple[int]] = 10000, random_state: int = 42,
                   noise: float = 0.05) -> Dataset:
    df: pd.DataFrame = pd.DataFrame(datasets.make_moons(
        n_samples=n_samples,
        noise=noise,
        random_state=random_state
    )[0])
    return create_dataset_from_df(df)


def generate_circles(n_samples: Union[int, Tuple[int]] = 10000, random_state: int = 42,
                     noise: float = 0.05) -> Dataset:
    df: pd.DataFrame = pd.DataFrame(datasets.make_circles(
        n_samples=n_samples,
        noise=noise,
        random_state=random_state
    )[0])
    return create_dataset_from_df(df)
