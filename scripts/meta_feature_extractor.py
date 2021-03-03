import numpy as np
import pandas as pd
from pymfe.mfe import MFE
from typing import Sequence

__default_mfe_params: dict = {
    "groups": "all",
    "summary": "all",
    "random_state": 42
}

def extract_from_object(dataset, mfe_params: dict = None) -> Sequence:
    if mfe_params is None:
        mfe_params = __default_mfe_params

    mfe = MFE(**mfe_params)
    mfe.fit(dataset)
    return mfe.extract()[1]

def extract_from_file(file: str, mfe_params: dict = None) -> Sequence:
    dataset = pd.read_csv(file)
    return extract_from_object(dataset)
