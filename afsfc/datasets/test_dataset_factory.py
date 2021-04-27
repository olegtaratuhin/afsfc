import pytest
from typing import Callable
import types
from .dataset_factory import extract_datasets
from .dataset import Dataset


class TestDatasetLoader:

    @pytest.mark.parametrize("config_path", ["../../data/real"])
    @pytest.mark.parametrize("cache_path", ["../../data/cache"])
    def test_loading_all_data(self, config_path, cache_path):
        dataset_source = extract_datasets(config_path, cache_path)
        assert dataset_source is not None
        assert isinstance(dataset_source, types.GeneratorType)
        try:
            assert next(dataset_source) is not None
        finally:
            pass
