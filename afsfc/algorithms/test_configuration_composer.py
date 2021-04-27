import pytest
from typing import List

from .config_space_composer import build_config_space, build_config_obj


class TestConfigurationSpaceComposer:

    @pytest.mark.parametrize("clustering_ls", [["KMeans", "DBSCAN"]])
    @pytest.mark.parametrize("feature_selection_ls", [["NullModel", "NormalizedCut"]])
    def test_should_return_composed_space(self, clustering_ls: List[str], feature_selection_ls: List[str]):
        cs = build_config_space(clustering_ls=clustering_ls, feature_selection_ls=feature_selection_ls)
        assert cs is not None
