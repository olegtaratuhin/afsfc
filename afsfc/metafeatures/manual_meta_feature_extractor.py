from typing import Callable

import numpy as np
import scipy.stats as sc_stat
import sklearn.decomposition
from sklearn.impute import SimpleImputer
from sklearn.metrics import mutual_info_score
from collections import Counter


class GeneralMetafeature(object):
    split_data_num = 1
    data_type = [['y_col', 'categorical_cols', 'numeric_cols', 'ordinal_cols']]

    @staticmethod
    def number_of_instances(x):
        return float(x.shape[0])

    @staticmethod
    def log_number_of_instances(x):
        return np.log(GeneralMetafeature.number_of_instances(x))

    @staticmethod
    def number_of_missing_values(x):
        return np.count_nonzero(x is None) + np.count_nonzero(x == '')

    @staticmethod
    def missing_values_ratio(x):
        return GeneralMetafeature.number_of_missing_values(x) / x.size

    @staticmethod
    def sparsity(x):
        return (np.count_nonzero(x == '') + np.count_nonzero(x == 0)) / x.size


class GeneralMetafeatureWithoutLabels(object):
    # dataset doesn't contain the label column
    split_data_num = 1
    data_type = [['categorical_cols', 'numeric_cols', 'ordinal_cols']]

    @staticmethod
    def number_of_features(x):
        return float(x.shape[1])

    @staticmethod
    def log_number_of_features(x):
        return np.log(GeneralMetafeatureWithoutLabels.number_of_features(x))

    @staticmethod
    def dataset_ratio(x):
        return float(GeneralMetafeatureWithoutLabels.number_of_features(x)) / \
               float(GeneralMetafeature.number_of_instances(x))

    @staticmethod
    def log_dataset_ratio(x):
        return np.log(GeneralMetafeatureWithoutLabels.dataset_ratio(x))


class LabelsMetafeatures(object):
    # dataset contains the label column only
    split_data_num = 1
    data_type = [['y_col']]

    @staticmethod
    def number_of_classes(x):
        return len(np.unique(x))

    @staticmethod
    def entropy_of_classes(x):
        x1 = x[x is not None]
        x1 = x1[x1 != '']
        freq_dict = Counter(x1)
        probs = np.array([value / len(x1) for value in freq_dict.values()])
        return np.sum(-np.log2(probs) * probs)


class NumericMetafeature(object):
    # dataset contains numeric columns only
    split_data_num = 1
    data_type = [['numeric_cols']]

    @staticmethod
    def sparsity_on_numeric_columns(x):
        return np.count_nonzero(x == 0) / x.size

    @staticmethod
    def _skewness(x, aggfunc: Callable):
        skewness = sc_stat.skew(x)
        # only finite values are accepted
        skewness = skewness[np.isfinite(skewness)]
        return aggfunc(skewness)

    @staticmethod
    def min_skewness(x):
        return NumericMetafeature._skewness(x, aggfunc=np.min)

    @staticmethod
    def max_skewness(x):
        return NumericMetafeature._skewness(x, aggfunc=np.max)

    @staticmethod
    def median_skewness(x):
        return NumericMetafeature._skewness(x, aggfunc=np.median)

    @staticmethod
    def mean_skewness(x):
        return NumericMetafeature._skewness(x, aggfunc=np.mean)

    @staticmethod
    def first_quartile_skewness(x):
        return NumericMetafeature._skewness(x, aggfunc=lambda s: np.percentile(s, 25))

    @staticmethod
    def third_quartile_skewness(x):
        return NumericMetafeature._skewness(x, aggfunc=lambda s: np.percentile(s, 75))

    @staticmethod
    def _kurtosis(x, aggfunc: Callable):
        kurtosis = sc_stat.kurtosis(x)
        kurtosis = kurtosis[np.isfinite(kurtosis)]
        return aggfunc(kurtosis)

    @staticmethod
    def min_kurtosis(x):
        return NumericMetafeature._kurtosis(x, aggfunc=np.min)

    @staticmethod
    def max_kurtosis(x):
        return NumericMetafeature._kurtosis(x, aggfunc=np.max)

    @staticmethod
    def median_kurtosis(x):
        return NumericMetafeature._kurtosis(x, aggfunc=np.median)

    @staticmethod
    def mean_kurtosis(x):
        return NumericMetafeature._kurtosis(x, aggfunc=np.mean)

    @staticmethod
    def first_quartile_kurtosis(x):
        return NumericMetafeature._kurtosis(x, aggfunc=lambda s: np.percentile(s, 25))

    @staticmethod
    def third_quartile_kurtosis(x):
        return NumericMetafeature._kurtosis(x, aggfunc=lambda s: np.percentile(s, 75))

    @staticmethod
    def _correlation(x, aggfunc: Callable):
        corr = np.corrcoef(x.T)
        corr = corr[np.isfinite(corr)]
        return aggfunc(corr)

    @staticmethod
    def min_correlation(x):
        return NumericMetafeature._correlation(x, aggfunc=np.min)

    @staticmethod
    def max_correlation(x):
        return NumericMetafeature._correlation(x, aggfunc=np.max)

    @staticmethod
    def _padded_correlation(x, aggfunc: Callable):
        corr = np.corrcoef(x.T)
        np.fill_diagonal(corr, np.nan)
        corr = corr.flatten()
        corr = corr[np.isfinite(corr)]
        return aggfunc(corr)

    @staticmethod
    def median_correlation(x):
        return NumericMetafeature._padded_correlation(x, aggfunc=np.median)

    @staticmethod
    def mean_correlation(x):
        return NumericMetafeature._padded_correlation(x, aggfunc=np.mean)

    @staticmethod
    def first_quartile_correlation(x):
        return NumericMetafeature._padded_correlation(x, aggfunc=lambda c: np.percentile(c, 25))

    @staticmethod
    def third_quartile_correlation(x):
        return NumericMetafeature._padded_correlation(x, aggfunc=lambda c: np.percentile(c, 75))

    @staticmethod
    def _covariance(x, aggfunc: Callable):
        cov = np.cov(x.T)
        cov = cov[np.isfinite(cov)]
        return aggfunc(cov)

    @staticmethod
    def min_covariance(x):
        return NumericMetafeature._covariance(x, aggfunc=np.min)

    @staticmethod
    def max_covariance(x):
        return NumericMetafeature._covariance(x, aggfunc=np.min)

    @staticmethod
    def median_covariance(x):
        return NumericMetafeature._covariance(x, aggfunc=np.median)

    @staticmethod
    def mean_covariance(x):
        return NumericMetafeature._covariance(x, aggfunc=np.mean)

    @staticmethod
    def first_quartile_covariance(x):
        return NumericMetafeature._covariance(x, aggfunc=lambda cov: np.percentile(cov, 25))

    @staticmethod
    def third_quartile_covariance(x):
        return NumericMetafeature._covariance(x, aggfunc=lambda cov: np.percentile(cov, 75))

    @staticmethod
    def pca_fraction_of_components_for_variance(x, variance=0.95, random_state=42):
        pca = sklearn.decomposition.PCA(copy=True)
        rs = np.random.RandomState(random_state)
        indices = np.arange(x.shape[0])

        # replace missing values using the mean along each column
        imp_mean = SimpleImputer(strategy='mean')
        x_transformed = imp_mean.fit_transform(x)

        for i in range(10):
            try:
                rs.shuffle(indices)
                pca.fit(x_transformed[indices])
                sum_ = 0.
                idx = 0
                while sum_ < variance and idx < len(pca.explained_variance_ratio_):
                    sum_ += pca.explained_variance_ratio_[idx]
                    idx += 1
                return float(idx) / float(x.shape[1])
            except Exception as e:
                pass

        print("Failed to compute PCA")
        # self.logger.warning("Failed to compute a Principle Component Analysis")
        return np.nan

    @staticmethod
    def pca_kurtosis_first_pc(x, strategy="mean", random_state=42):
        pca = sklearn.decomposition.PCA(copy=True)
        rs = np.random.RandomState(random_state)
        indices = np.arange(x.shape[0])

        imp_mean = SimpleImputer(strategy=strategy)
        x_transformed = imp_mean.fit_transform(x)

        for i in range(10):
            try:
                rs.shuffle(indices)
                pca.fit(x_transformed[indices])
                components = pca.components_
                pca.components_ = components[:1]
                transformed = pca.transform(x_transformed)
                pca.components_ = components
                kurtosis = sc_stat.kurtosis(transformed)
                return kurtosis[0]
            except Exception as e:
                pass

        print("Failed to compute PCA")
        # self.logger.warning("Failed to compute a Principle Component Analysis")
        return np.nan

    @staticmethod
    def pca_skewness_first_pc(X):
        pca = sklearn.decomposition.PCA(copy=True)
        rs = np.random.RandomState(42)
        indices = np.arange(X.shape[0])

        imp_mean = SimpleImputer(strategy='mean')
        x_transformed = imp_mean.fit_transform(X)

        for i in range(10):
            try:
                rs.shuffle(indices)
                pca.fit(x_transformed[indices])
                components = pca.components_
                pca.components_ = components[:1]
                transformed = pca.transform(x_transformed)
                pca.components_ = components
                skewness = sc_stat.skew(transformed)
                return skewness[0]

            except Exception as e:
                pass

        print("Failed to compute PCA")
        # self.logger.warning("Failed to compute a Principle Component Analysis")
        return np.nan


class CategoricalMetafeature(object):
    # dataset contains categorical columns only
    split_data_num = 1
    data_type = [['categorical_cols', 'ordinal_cols']]

    @staticmethod
    def _entropy(x, aggfunc: Callable):
        entropy = []
        for sublist in x.T.astype(str):
            sublist1 = sublist[sublist is not None]
            sublist1 = sublist1[sublist1 != '']
            freq_dict = Counter(sublist1)
            probs = np.array([value / len(sublist1) for value in freq_dict.values()])
            entropy.append(np.sum(-np.log2(probs) * probs))
        return aggfunc(entropy) / np.log2(GeneralMetafeature.number_of_instances(x))

    @staticmethod
    def min_entropy(x):
        return CategoricalMetafeature._entropy(x, aggfunc=np.min)

    @staticmethod
    def max_entropy(x):
        return CategoricalMetafeature._entropy(x, aggfunc=np.max)

    @staticmethod
    def median_entropy(x):
        return CategoricalMetafeature._entropy(x, aggfunc=np.median)

    @staticmethod
    def mean_entropy(x):
        return CategoricalMetafeature._entropy(x, aggfunc=np.mean)

    @staticmethod
    def first_quartile_entropy(x):
        return CategoricalMetafeature._entropy(x, aggfunc=lambda q: np.percentile(q, 25))

    @staticmethod
    def third_quartile_entropy(x):
        return CategoricalMetafeature._entropy(x, aggfunc=lambda q: np.percentile(q, 75))


class CategoricalMetafeatureWithLabels(object):
    # dataset contains categorical columns and the label column
    split_data_num = 2
    data_type = [['y_col'], ['categorical_cols', 'ordinal_cols']]

    @staticmethod
    def _mutual_information(class_col, cat_cols, aggfunc: Callable):
        x = np.concatenate((class_col, cat_cols), axis=1).astype(str)
        x = x[~(x == '').any(axis=1)]
        x = x[~(x is None).any(axis=1)]

        mut_inf = []
        class_col1 = x[:, 0]
        for sublist in x.T[1:]:
            mut_inf.append(mutual_info_score(labels_true=class_col1, labels_pred=sublist))

        return aggfunc(mut_inf)

    @staticmethod
    def min_mutual_information(class_col, cat_cols):
        return CategoricalMetafeatureWithLabels._mutual_information(class_col, cat_cols, aggfunc=np.min)

    # class_col has the label column only, cat_cols has categorical columns only
    @staticmethod
    def max_mutual_information(class_col, cat_cols):
        return CategoricalMetafeatureWithLabels._mutual_information(class_col, cat_cols, aggfunc=np.max)

    # class_col has the label column only, cat_cols has categorical columns only
    @staticmethod
    def median_mutual_information(class_col, cat_cols):
        return CategoricalMetafeatureWithLabels._mutual_information(class_col, cat_cols, aggfunc=np.median)

    # class_col has the label column only, cat_cols has categorical columns only
    @staticmethod
    def mean_mutual_information(class_col, cat_cols):
        return CategoricalMetafeatureWithLabels._mutual_information(class_col, cat_cols, aggfunc=np.mean)

    # class_col has the label column only, cat_cols has categorical columns only
    @staticmethod
    def first_quartile_mutual_information(class_col, cat_cols):
        return CategoricalMetafeatureWithLabels._mutual_information(class_col, cat_cols,
                                                                    aggfunc=lambda x: np.percentile(x, 25))

    # class_col has the label column only, cat_cols has categorical columns only
    @staticmethod
    def third_quartile_mutual_information(class_col, cat_cols):
        return CategoricalMetafeatureWithLabels._mutual_information(class_col, cat_cols,
                                                                    aggfunc=lambda x: np.percentile(x, 75))


class MetafeatureMapper(object):
    feature_type = {
        f: GeneralMetafeature for f in list(GeneralMetafeature.__dict__) if '_' not in f
    }
    feature_type.update(
        {f: GeneralMetafeatureWithoutLabels for f in list(GeneralMetafeatureWithoutLabels.__dict__) if '_' not in f})
    feature_type.update({f: NumericMetafeature for f in list(NumericMetafeature.__dict__) if '_' not in f})
    feature_type.update({f: CategoricalMetafeature for f in list(CategoricalMetafeature.__dict__) if '_' not in f})
    feature_type.update(
        {f: CategoricalMetafeatureWithLabels for f in list(CategoricalMetafeatureWithLabels.__dict__) if '_' not in f})

    feature_function = {
        f_name: f_obj for f_name, f_obj in GeneralMetafeature.__dict__.items() if '_' not in f_name
    }
    feature_function.update(
        {f_name: f_obj for f_name, f_obj in GeneralMetafeatureWithoutLabels.__dict__.items() if '_' not in f_name})
    feature_function.update(
        {f_name: f_obj for f_name, f_obj in NumericMetafeature.__dict__.items() if '_' not in f_name})
    feature_function.update(
        {f_name: f_obj for f_name, f_obj in CategoricalMetafeature.__dict__.items() if '_' not in f_name})
    feature_function.update(
        {f_name: f_obj for f_name, f_obj in CategoricalMetafeatureWithLabels.__dict__.items() if '_' not in f_name})

    @staticmethod
    def get_class(string):
        return MetafeatureMapper.feature_type.get(string, None)

    @staticmethod
    def get_metafeature_function(string):
        return MetafeatureMapper.feature_function.get(string, None)

    @staticmethod
    def get_all_metafeatures():
        return list(MetafeatureMapper.feature_type.keys())

    @staticmethod
    def get_general_metafeatures():
        return [key for key, value in MetafeatureMapper.feature_type.items() if value is GeneralMetafeature]

    @staticmethod
    def get_general_metafeatures_without_labels():
        return [key for key, value in MetafeatureMapper.feature_type.items() if
                value is GeneralMetafeatureWithoutLabels]

    @staticmethod
    def get_numeric_metafeatures():
        return [key for key, value in MetafeatureMapper.feature_type.items() if value is NumericMetafeature]

    @staticmethod
    def get_categorical_metafeatures():
        return [key for key, value in MetafeatureMapper.feature_type.items() if value is CategoricalMetafeature]

    @staticmethod
    def get_categorical_metafeatures_with_labels():
        return [key for key, value in MetafeatureMapper.feature_type.items() if
                value is CategoricalMetafeatureWithLabels]


def calculate_metafeatures(raw_dataset, file_dict, metafeature_ls=None):
    if metafeature_ls is None:
        metafeature_ls = []
    if len(metafeature_ls) == 0:
        metafeature_ls = MetafeatureMapper.get_all_metafeatures()

    values = []

    for feature_str in metafeature_ls:
        feature_class = MetafeatureMapper.get_class(feature_str)
        datasets = []
        app_data_type = True

        for i in range(feature_class.splitted_data_num):
            col_list = []

            for col in feature_class.data_type[i]:
                if file_dict[col]:
                    temp_data = raw_dataset[file_dict[col]].to_numpy()
                    if temp_data.ndim == 1:
                        temp_data = np.reshape(temp_data, (-1, 1))
                    col_list.append(temp_data)

            if len(col_list) == 0:
                values.append(None)
                app_data_type = False
                break
            else:
                datasets.append(np.concatenate(tuple(col_list), axis=1))

        if app_data_type:
            values.append(MetafeatureMapper.get_metafeature_function(feature_str).__get__(object)(*datasets))

    return np.reshape(np.array(values), (1, -1))
