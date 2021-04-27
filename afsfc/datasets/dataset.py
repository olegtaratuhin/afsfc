# Generated source file

from typing import List, Optional, Dict, Any, TypeVar, Callable, Type, cast
import pandas as pd


T = TypeVar("T")


def from_list(f: Callable[[Any], T], x: Any) -> List[T]:
    assert isinstance(x, list)
    return [f(y) for y in x]


def from_str(x: Any) -> str:
    assert isinstance(x, str)
    return x


def from_bool(x: Any) -> bool:
    assert isinstance(x, bool)
    return x


def from_none(x: Any) -> Any:
    assert x is None
    return x


def from_union(fs, x):
    for f in fs:
        try:
            return f(x)
        except:
            pass
    assert False


def from_dict(f: Callable[[Any], T], x: Any) -> Dict[str, T]:
    assert isinstance(x, dict)
    return { k: f(v) for (k, v) in x.items() }


def to_class(c: Type[T], x: Any) -> dict:
    assert isinstance(x, c)
    return cast(Any, x).to_dict()


class Dataset:
    """Dataset meta information"""
    """Array of categorical feature names"""
    categorical: List[str]
    """Flag to indicate that default task for this dataset is classification"""
    classification: Optional[bool]
    """Array of feature names to ignore"""
    ignore: List[str]
    """Array of numerical feature names"""
    numerical: List[str]
    """Array of ordinal feature names"""
    ordinal: Dict[str, List[str]]
    """Path to dataset on disk, might be null in case it is not downloaded"""
    path: str
    """Name of the target feature"""
    target: Optional[str]
    """openML dataset path"""
    url: Optional[str]
    """Pandas dataframe with dataset content"""
    content: pd.DataFrame

    def __init__(self, categorical: List[str], classification: Optional[bool], ignore: List[str], numerical: List[str], ordinal: Dict[str, List[str]], path: str, target: Optional[str], url: Optional[str]) -> None:
        self.categorical = categorical
        self.classification = classification
        self.ignore = ignore
        self.numerical = numerical
        self.ordinal = ordinal
        self.path = path
        self.target = target
        self.url = url

    @staticmethod
    def from_dict(obj: Any) -> 'Dataset':
        assert isinstance(obj, dict)
        categorical = from_list(from_str, obj.get("categorical"))
        classification = from_union([from_bool, from_none], obj.get("classification"))
        ignore = from_list(from_str, obj.get("ignore"))
        numerical = from_list(from_str, obj.get("numerical"))
        ordinal = from_dict(lambda x: from_list(from_str, x), obj.get("ordinal"))
        path = from_str(obj.get("path"))
        target = from_union([from_str, from_none], obj.get("target"))
        url = from_union([from_str, from_none], obj.get("url"))
        return Dataset(categorical, classification, ignore, numerical, ordinal, path, target, url)

    def to_dict(self) -> dict:
        result: dict = {}
        result["categorical"] = from_list(from_str, self.categorical)
        result["classification"] = from_union([from_bool, from_none], self.classification)
        result["ignore"] = from_list(from_str, self.ignore)
        result["numerical"] = from_list(from_str, self.numerical)
        result["ordinal"] = from_dict(lambda x: from_list(from_str, x), self.ordinal)
        result["path"] = from_str(self.path)
        result["target"] = from_union([from_str, from_none], self.target)
        result["url"] = from_union([from_str, from_none], self.url)
        return result


def dataset_from_dict(s: Any) -> Dataset:
    return Dataset.from_dict(s)


def dataset_to_dict(x: Dataset) -> Any:
    return to_class(Dataset, x)
