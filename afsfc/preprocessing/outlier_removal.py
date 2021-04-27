import pandas as pd
from sklearn.ensemble import IsolationForest
from afsfc.datasets import Dataset
from abc import ABC


class AbstractOutlierRemoval(ABC):
    """
    Abstract base class for outlier removal module
    """
    def remove_outliers(self, dataset: Dataset) -> Dataset:
        raise NotImplemented()


class DummyOutlierRemoval(AbstractOutlierRemoval):
    """
    Dummy outlier removal does nothing and returns original dataset
    """
    def remove_outliers(self, dataset: Dataset) -> Dataset:
        return dataset


class IsolationForestOutlierRemoval(AbstractOutlierRemoval):
    """
    Isolation forest outlier removal uses ensemble of trees to determine possible outliers
    """
    def remove_outliers(self, dataset: Dataset) -> Dataset:
        return DummyOutlierRemoval().remove_outliers(dataset)


class OneClassSVMOutlierRemoval(AbstractOutlierRemoval):
    """
    Experiment results:
    One class SVM is known to be sensitive to outliers and thus doesn't perform very well for
    outlier detection in high-dimension, or without any assumptions on the distribution on the inlaying data.
    May still be used but requires fine-tuning of it's hyperparameter 'nu'.
    """

    def remove_outliers(self, dataset: Dataset) -> Dataset:
        return DummyOutlierRemoval().remove_outliers(dataset)
