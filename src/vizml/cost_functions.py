from abc import ABC, abstractmethod
from typing import Sequence, Union, Any

from numpy.typing import NDArray
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error


class BaseCostFunction(ABC):
    """Base class for all cost functions."""

    @abstractmethod
    def compute(self, array1: Union[NDArray[Any], Sequence[Any]],
                array2: Union[NDArray[Any], Sequence[Any]]):
        """Computes the cost."""


class MeanSquaredError(BaseCostFunction):
    """Class to compute the cost using Mean Square Error."""

    def compute(self, array1: Union[NDArray[Any], Sequence[Any]],
                array2: Union[NDArray[Any], Sequence[Any]]):
        return mean_squared_error(array1, array2)


class RootMeanSquaredError(BaseCostFunction):
    """Class to compute the cost using Root Mean Square Error."""

    def compute(self, array1: Union[NDArray[Any], Sequence[Any]],
                array2: Union[NDArray[Any], Sequence[Any]]):
        return mean_squared_error(array1, array2, squared=False)


class MeanAbsoluteError(BaseCostFunction):
    """Class to compute the cost using Mean Absolute Error."""

    def compute(self, array1: Union[NDArray[Any], Sequence[Any]],
                array2: Union[NDArray[Any], Sequence[Any]]):
        return mean_absolute_error(array1, array2)


class MaxError(BaseCostFunction):
    """Class to compute the cost using the Maximum Error."""

    def compute(self, array1: Union[NDArray[Any], Sequence[Any]],
                array2: Union[NDArray[Any], Sequence[Any]]):
        return max_error(array1, array2)
