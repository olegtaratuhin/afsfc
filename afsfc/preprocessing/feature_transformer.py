import itertools
from typing import Callable
import pandas as pd
from afsfc.datasets import Dataset
from abc import ABC
from sklearn.preprocessing import StandardScaler


class AbstractFeatureTransformer(ABC):
    """
    Abstract feature transformer interface class
    """

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Performs some transformation on the dataset and returns modified dataset
        Args:
            dataset (Dataset): dataset to transform

        Returns:
            Dataset after transformation
        """
        pass


class FeatureTransformer(AbstractFeatureTransformer):
    """
    Transformer that does manipulations on feature level
    """

    @staticmethod
    def _transform_ignored(dataset: Dataset) -> Dataset:
        """
        Drop ignored features as specified in schema
        Args:
            dataset (Dataset): input dataset

        Returns:
            Dataset after transformation
        """
        dataset.content.dropna(inplace=True)
        if len(dataset.ignore) == 0:
            return dataset

        dataset.content.drop(columns=dataset.ignore, inplace=True)
        return dataset

    @staticmethod
    def _transform_categorical(dataset: Dataset) -> Dataset:
        """
        Transform categorical features to one-hot-encoding
        Args:
            dataset (Dataset): input dataset

        Returns:
            Dataset after transformation
        """
        content, categorical_features = dataset.content, dataset.categorical
        if len(categorical_features) == 0:
            return dataset

        encodings, encoded_cols = {}, {}

        for feature in categorical_features:
            content[feature] = content[feature].astype('category')
            encodings[feature] = dict(enumerate(content[feature].cat.categories))
            integer_encoded_col = content[feature].cat.codes
            one_hot_encoded_col = pd.get_dummies(integer_encoded_col, prefix=feature, drop_first=True)
            encoded_cols[feature] = one_hot_encoded_col

        ordinal_and_numerical = itertools.chain(dataset.numerical, dataset.ordinal)
        encoded_dataset = pd.DataFrame(content[ordinal_and_numerical])
        for col in encoded_cols:
            encoded_dataset[encoded_cols[col].columns] = encoded_cols[col]
        dataset.content = encoded_dataset

        return dataset

    @staticmethod
    def _transform_ordinal(dataset: Dataset) -> Dataset:
        """
        Transform ordinal features by enumerating them
        Args:
            dataset (Dataset): input dataset

        Returns:
            Dataset after transformation
        """
        content, ordinal_features = dataset.content, dataset.ordinal
        if len(ordinal_features) == 0:
            return dataset
        encodings, encoded_cols = {}, {}

        for feature in ordinal_features:
            encodings[feature] = dict(zip(ordinal_features[feature], range(len(ordinal_features[feature]))))
            encoded_cols[feature] = pd.DataFrame(content[feature]
                                                 .replace(to_replace=encodings[feature]), columns=[feature])

        encoded_dataset = pd.DataFrame(content[dataset.numerical])
        for col in encoded_cols:
            encoded_dataset[encoded_cols[col].columns] = encoded_cols[col]
        dataset.content = encoded_dataset

        return dataset

    @staticmethod
    def _truncate_dataset(dataset: Dataset, threshold: int = 10000) -> Dataset:
        """
        Truncate dataset to specified number of rows
        As pretraining may take a long time we may reduce dataset size so that processing takes less time
        but convergence doesn't suffer too much.

        Args:
            dataset (Dataset): input dataset
            threshold (int): maximum desired value of dataset length in rows

        Returns:
            Dataset after transformation
        """
        instance_count = len(dataset.content.index)
        if instance_count >= threshold:
            dataset.content = dataset.content[:threshold]
        return dataset

    @staticmethod
    def _transform_drop_target(dataset: Dataset) -> Dataset:
        """
        Drop target from dataset
        Args:
            dataset (Dataset): input dataset

        Returns:
            Dataset after transformation
        """
        target_name = dataset.target
        if target_name is not None:
            dataset.content.drop(columns=[target_name], inplace=True)
        return dataset

    @staticmethod
    def _scale_features(dataset: Dataset) -> Dataset:
        scaler = StandardScaler()
        df = dataset.content
        dataset.content[df.columns] = scaler.fit_transform(df[df.columns])
        return dataset

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Pipes all transformations together and returns dataset after all modifications
        Args:
            dataset (Dataset): input dataset

        Returns:
            Dataset after transformation
        """
        pipeline: [Callable[[Dataset], Dataset]] = [
            FeatureTransformer._transform_ignored,
            FeatureTransformer._transform_drop_target,
            FeatureTransformer._transform_categorical,
            FeatureTransformer._transform_ordinal,
            FeatureTransformer._truncate_dataset,
            FeatureTransformer._scale_features
        ]

        for transformation in pipeline:
            dataset = transformation(dataset)

        return dataset
