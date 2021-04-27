import pathlib
import os
from typing import Generator, TextIO, Union
import json
from .dataset import Dataset
import pandas as pd
import logging
import sys


def __list_config_files(config_path: str) -> Generator[str, None, None]:
    for root, _, files in os.walk(config_path, topdown=True):
        for file in files:
            if not file.endswith(".json"):
                _dataset_factory_logger.debug(f"Skipping non-config file: {file}")
                continue
            yield f"{root}/{file}"
        
        # we only go one level
        break


def load_dataset(config_file: Union[TextIO, str], cache_path: str) -> Dataset:
    _dataset_factory_logger.debug(f"loading dataset from: {cache_path}")
    json_config: dict
    if isinstance(config_file, str):
        with open(config_file) as f:
            json_config = json.load(f)
    else:
        json_config = json.load(config_file)
    dataset: Dataset = Dataset.from_dict(json_config)
    dataset.content = pd.read_csv(f"{cache_path}/{dataset.path}")
    return dataset


def extract_datasets(config_path: str, cache_path: str) -> Generator[Dataset, None, None]:
    assert config_path is not None
    assert cache_path is not None

    for config_file in __list_config_files(config_path):
        _dataset_factory_logger.info(f"Found {config_file}")
        with open(config_file) as config:
            yield load_dataset(config, cache_path)


_dataset_factory_logger = logging.getLogger(__file__)
