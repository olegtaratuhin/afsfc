from afsfc.datasets import Dataset, load_dataset
from .feature_transformer import FeatureTransformer


class TestFeatureTransformer:

    def test_sample_dataset(self):
        test_dataset: Dataset = load_dataset("../data/test_sample/weatherAUS.json", "../data/cache")

        assert test_dataset.categorical is not None and len(test_dataset.categorical) > 0

        original_features = list(test_dataset.content.columns)

        transformer = FeatureTransformer()
        transformer.transform(test_dataset)

        transformed_features = list(test_dataset.content)

        # pytest magic
        assert original_features != transformed_features
