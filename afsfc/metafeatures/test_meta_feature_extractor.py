import pytest
from typing import Callable
from sklearn.datasets import load_iris, load_boston, load_breast_cancer
from sklearn.utils import Bunch

from .meta_feature_extractor import *


class TestFeatureExtractor:

    @pytest.mark.parametrize("dataset", [load_iris, load_boston, load_breast_cancer])
    def test_should_return_vector(self, dataset: Callable[..., Bunch]):
        data = dataset().data
        vec = extract_from_object(data)
        assert vec is not None
        assert hasattr(vec, '__len__')
        assert len(vec) > 0
