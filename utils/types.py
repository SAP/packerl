"""
This file redefines various types to increase clarity.
"""
from typing import Dict, Any, List, Union, Iterable, Callable, Optional, Tuple, Generator, Type, Set, Type, Literal, \
    TYPE_CHECKING
import copy
from functools import partial

import networkx as nx
from numpy import ndarray
from torch import Tensor
from torch_geometric.data.batch import Batch
from torch_geometric.data.hetero_data import HeteroData
from torch_geometric.data.data import Data, BaseData


Key = Union[str, int]  # for dictionaries, we usually use strings or ints as keys
ConfigDict = Dict[Key, Any]  # A (potentially nested) dictionary containing the "params" section of the .yaml file
EntityDict = Dict[Key, Union[Dict, str]]  # potentially nested dictionary of entities
ValueDict = Dict[Key, Any]
Result = Union[List, int, float, ndarray]
Shape = Union[int, Iterable, ndarray]

InputBatch = Union[Dict[Key, Tensor], Tensor, Batch, Data, HeteroData, None]
OutputTensorDict = Dict[Key, Tensor]

GlobalMonitoring = nx.DiGraph
LocalMonitoring = Dict[Any, nx.DiGraph]

GlobalReward = Tensor
PerAgentReward = Tensor
RewardDict = Dict[str, Union[float, Dict[str, float]]]
