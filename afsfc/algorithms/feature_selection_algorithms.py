from afsfc.algorithms import fsfc
from numpy import prod
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, OrdinalHyperparameter


class FeatureSelectionAlgorithms(object):

    class Metaclass(type):

        @property
        def name(cls):
            return cls._name

        @property
        def model(cls):
            return cls._model

        @property
        def params(cls):
            return cls._params

        @property
        def params_names(cls):
            return cls._params_names

        @property
        def conditions(cls):
            return cls._conditions

        @property
        def forbidden_clauses(cls):
            return cls._forbidden_clauses

        @property
        def has_discrete_cfg_space(cls):
            is_discrete = lambda param: isinstance(param, UniformIntegerHyperparameter) or \
                                        isinstance(param, OrdinalHyperparameter) or \
                                        isinstance(param, CategoricalHyperparameter)
            return all([is_discrete(param) for param in cls._params])

        @property
        def n_possible_cfgs(cls):
            if not cls.has_discrete_cfg_space:
                return float('inf')
            else:
                def n_possible_values(param):
                    if isinstance(param, CategoricalHyperparameter):
                        return len(param.choices)
                    elif isinstance(param, OrdinalHyperparameter):
                        return len(param.sequence)
                    elif isinstance(param, UniformIntegerHyperparameter):
                        return param.upper - param.lower + 1

                return prod([n_possible_values(param) for param in cls._params])

    class Lasso(object, metaclass=Metaclass):
        __doc__ = fsfc.Lasso.__doc__
        _name = "Lasso"
        _model = fsfc.Lasso
        _params = [
            UniformIntegerHyperparameter("k", 3, 100, default_value=20),
            UniformFloatHyperparameter("norm_constraint", 0.01, 10.0)
        ]
        _params_names = set([p.name for p in _params])
        _conditions = []
        _forbidden_clauses = []

    class LFSBSS(object, metaclass=Metaclass):
        _name = "LFSBSS"
        _model = fsfc.LFSBSS
        _params = [
            UniformIntegerHyperparameter("clusters", 2, 10, default_value=3)
        ]
        _params_names = set([p.name for p in _params])
        _conditions = []
        _forbidden_clauses = []

    class MCFS(object, metaclass=Metaclass):
        _name = "MCFS"
        _model = fsfc.MCFS
        _params = [
            UniformIntegerHyperparameter("k", 3, 100, default_value=20),
            UniformIntegerHyperparameter("clusters", 2, 10, default_value=3),
            UniformIntegerHyperparameter("p", 3, 15, default_value=8),
            UniformIntegerHyperparameter("sigma", 1, 3, default_value=1),
            # CategoricalHyperparameter("mode", ["default", "lasso"], default_value="default"),
            # UniformFloatHyperparameter("alpha", 0.0001, 1.0, default_value=0.01)
        ]
        _params_names = set([p.name for p in _params])
        _conditions = []
        _forbidden_clauses = []

    class WKMeans(object, metaclass=Metaclass):
        _name = "WKMeans"
        _model = fsfc.WKMeans
        _params = [
            UniformIntegerHyperparameter("k", 3, 100, default_value=20),
            UniformFloatHyperparameter("beta", 0.001, 10.0, default_value=1.0)
        ]
        _params_names = set([p.name for p in _params])
        _conditions = []
        _forbidden_clauses = []

    class GenericSPEC(object, metaclass=Metaclass):
        _name = "GenericSPEC"
        _model = fsfc.GenericSPEC
        _params = [
            UniformIntegerHyperparameter("k", 3, 100, default_value=20)
        ]
        _params_names = set([p.name for p in _params])
        _conditions = []
        _forbidden_clauses = []

    class FixedSPEC(object, metaclass=Metaclass):
        _name = "FixedSPEC"
        _model = fsfc.FixedSPEC
        _params = [
            UniformIntegerHyperparameter("k", 3, 100, default_value=20),
            UniformIntegerHyperparameter("clusters", 2, 10, default_value=3),
        ]
        _params_names = set([p.name for p in _params])
        _conditions = []
        _forbidden_clauses = []

    class NormalizedCut(object, metaclass=Metaclass):
        _name = "NormalizedCut"
        _model = fsfc.NormalizedCut
        _params = [
            UniformIntegerHyperparameter("k", 3, 100, default_value=20)
        ]
        _params_names = set([p.name for p in _params])
        _conditions = []
        _forbidden_clauses = []

    class NullModel(object, metaclass=Metaclass):
        """
        Dummy model, no feature selection is performed
        """
        class _model(object):
            def __init__(self):
                pass

            def fit_transform(self, data):
                return data

            def transform(self, data):
                return data

        _name = "NullModel"
        _model = _model
        _params = []
        _params_names = set([p.name for p in _params])
        _conditions = []
        _forbidden_clauses = []
