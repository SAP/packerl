import abc
from typing import Dict, Optional, Tuple, Union

from torch_geometric.data.data import BaseData

from rl.nn.hmpn.common.hmpn_util import noop


class AbstractGraphAssertions(abc.ABC):
    """
    Asserts that the input graph has the correct shape.
    """
    def __init__(self, *,
                 in_node_features: Union[int, Dict[str, int]],
                 in_edge_features: Union[int, Dict[Tuple[str, str, str], int]],
                 in_global_features: Optional[int]
                 ):
        """

        Args:
            in_node_features: The number of input node features (per node type)
            in_edge_features: The number of input edge features (per edge type)
            in_global_features: The number of input global features, None if no global features are used
        """
        self._assertion_dict = {"in_node_features": in_node_features,
                                "in_edge_features": in_edge_features,
                                "in_global_features": in_global_features
                                }

        if in_global_features is not None:
            self._maybe_global_assertions = self._global_assertions
            self._assert_global = True
        else:
            self._maybe_global_assertions = noop
            self._assert_global = False

    def _global_assertions(self, graph: BaseData):
        """
        Asserts that the global features are of the correct shape
        Args:
            graph: (batch of) heterogeneous or homogeneous graph(s)

        Returns: None

        """
        in_global_features = self._assertion_dict.get("in_global_features")
        assert hasattr(graph, "u"), "Graph does not provide global features"
        assert in_global_features == graph.u.shape[-1], "Global feature dimensions do not match. Given" \
                                                         f"'{graph.u.shape[-1]}', expected '{in_global_features}'"

    def __call__(self, tensor: BaseData):
        """
        Does various shape assertions to make sure that the (batch of) graph(s) is built correctly
        Args:
            tensor: (batch of) heterogeneous graph(s)

        Returns:

        """
        self._maybe_global_assertions(tensor)

