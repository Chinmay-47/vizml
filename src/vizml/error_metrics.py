from abc import ABC, abstractmethod
from typing import Sequence, Union, Any

from numpy.typing import NDArray
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error


class BaseErrorMetric(ABC):
    """Base class for all error metrics."""

    @abstractmethod
    def compute(self, array1: Union[NDArray[Any], Sequence[Any]],
                array2: Union[NDArray[Any], Sequence[Any]]):
        """Computes the cost."""


class MeanSquaredError(BaseErrorMetric):
    """Class to compute the error using Mean Square Error."""

    def compute(self, array1: Union[NDArray[Any], Sequence[Any]],
                array2: Union[NDArray[Any], Sequence[Any]]):
        return mean_squared_error(array1, array2)


class RootMeanSquaredError(BaseErrorMetric):
    """Class to compute the error using Root Mean Square Error."""

    def compute(self, array1: Union[NDArray[Any], Sequence[Any]],
                array2: Union[NDArray[Any], Sequence[Any]]):
        return mean_squared_error(array1, array2, squared=False)


class MeanAbsoluteError(BaseErrorMetric):
    """Class to compute the error using Mean Absolute Error."""

    def compute(self, array1: Union[NDArray[Any], Sequence[Any]],
                array2: Union[NDArray[Any], Sequence[Any]]):
        return mean_absolute_error(array1, array2)


class MaxError(BaseErrorMetric):
    """Class to compute the error using the Maximum Error."""

    def compute(self, array1: Union[NDArray[Any], Sequence[Any]],
                array2: Union[NDArray[Any], Sequence[Any]]):
        return max_error(array1, array2)


class Errors:
    """Class to enumerate all available error metrics."""

    MeanSquared = MeanSquaredError()
    RootMeanSquared = RootMeanSquaredError()
    MeanAbsolute = MeanAbsoluteError()
    Max = MaxError()
