import itertools

from typing import List

from afsfc.algorithms.clustering_algorithms import ClusteringAlgorithms
from afsfc.algorithms.feature_selection_algorithms import FeatureSelectionAlgorithms
from afsfc.utils.string_utils import encode_parameter

from smac.configspace import ConfigurationSpace, Configuration
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from ConfigSpace.conditions import InCondition


class Mapper(object):
    __combined_algorithms_items = {
        **dict(ClusteringAlgorithms.__dict__.items()),
        **dict(FeatureSelectionAlgorithms.__dict__.items())
    }

    d = {
        class_name: class_obj for class_name, class_obj in __combined_algorithms_items.items()
        if '_' not in class_name
    }

    @staticmethod
    def get_class(string):
        return Mapper.d.get(string, None)

    @staticmethod
    def get_algorithms():
        return list(Mapper.d.keys())


def build_config_space(clustering_ls: List[str] = None, feature_selection_ls: List[str] = None):
    if clustering_ls is None:
        clustering_ls = ["KMeans", "DBSCAN"]
    if feature_selection_ls is None:
        feature_selection_ls = ["NullModel", "NormalizedCut"]
    assert len(clustering_ls) > 0
    assert len(feature_selection_ls) > 0

    cs = ConfigurationSpace()

    clustering_choice = CategoricalHyperparameter("clustering_choice",
                                                  clustering_ls,
                                                  default_value=clustering_ls[0])
    feature_selection_choice = CategoricalHyperparameter("feature_selection_choice",
                                                         feature_selection_ls,
                                                         default_value=feature_selection_ls[0])
    cs.add_hyperparameters([clustering_choice])
    cs.add_hyperparameters([feature_selection_choice])

    for idx, string in enumerate(itertools.chain(clustering_ls, feature_selection_ls)):
        algorithm = Mapper.get_class(string)

        for param in algorithm.params:
            encoded_string = encode_parameter(param.name, algorithm.name)
            param.name = encoded_string

        cs.add_hyperparameters(algorithm.params)

        for param in algorithm.params:
            cs.add_condition(InCondition(
                child=param,
                parent=clustering_choice if idx < len(clustering_ls) else feature_selection_choice,
                values=[string]
            ))

        for condition in algorithm.forbidden_clauses:
            cs.add_forbidden_clause(condition)

    return cs


def build_config_obj(config_space: ConfigurationSpace, values_dict: dict):
    unconditional_parameters = config_space.get_all_unconditional_hyperparameters()

    for hyperparam in unconditional_parameters:
        choice = values_dict.get(hyperparam, None)
        if choice is None:
            continue

        algorithm = Mapper.get_class(choice)
        for param in algorithm.params:
            param_name = encode_parameter(param.name, algorithm.name)
            if param_name not in values_dict:
                values_dict[param_name] = param.default_value

    return Configuration(configuration_space=config_space, values=values_dict)
