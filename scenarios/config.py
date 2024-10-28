from typing import TypeVar, List, Union, Dict, Any, Generic, get_origin, get_args

import numpy as np
from pydantic import BaseModel, ConfigDict

T = TypeVar('T')


class ChooseFromList(List[T], Generic[T]):
    """
    A list of elements of type T. When sampling from this list, a random element is chosen.
    """
    pass


SingleOrList = Union[T, List[T]]
SingleOrChoice = Union[T, ChooseFromList[T]]


def is_generic_single_or_choice_type(typ) -> bool:
    """
    Check if the provided type `typ` is a SingleOrChoice type for any T.
    """
    # Get the origin of the type (should be Union for SingleOrChoice)
    if get_origin(typ) is not Union:
        return False

    # Get the arguments of the type
    args = get_args(typ)
    if len(args) != 2:
        return False

    # Check if one of the arguments is ChooseFromList with any type
    if get_origin(args[1]) != ChooseFromList:
        # If not, maybe the order is reversed
        if get_origin(args[0]) != ChooseFromList:
            return False

    return True


def _sample_field(value, rng):
    """
    Sample a field value using the provided RNG.
    """
    if isinstance(value, BaseConfig):
        return value.sample(rng)
    elif isinstance(value, ChooseFromList):
        return rng.choice(value)  # Sample from the list using the provided RNG
    elif isinstance(value, dict):
        return {key: _sample_field(val, rng) for key, val in value.items()}
    else:
        return value  # Return the value directly


class BaseConfig(BaseModel):
    """
    Base configuration class that supports sampling and ChooseFromList.
    It is used throughout the scenarios to define configuration classes.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        """
        Initialize the configuration object. If a field is of type list, it is converted to a ChooseFromList
        """
        for field_name, field_type in self.__annotations__.items():
            if is_generic_single_or_choice_type(field_type):
                field_data = data.get(field_name, [])
                if isinstance(field_data, list):
                    data[field_name] = ChooseFromList(field_data)
        super().__init__(**data)

    def sample(self, rng: np.random.Generator) -> Dict[str, Any]:
        """
        Sample the configuration using the provided RNG. This is relevant for ChooseFromList fields.
        """
        return {name: _sample_field(value, rng) for name, value in self.__dict__.items()}
