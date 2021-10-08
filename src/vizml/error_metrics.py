from abc import ABC, abstractmethod
from typing import Sequence, Union, Any, List, Tuple

from numpy.typing import NDArray
from sklearn.metrics import (mean_squared_error, mean_absolute_error, max_error, mean_squared_log_error,
                             median_absolute_error, mean_absolute_percentage_error)


class BaseErrorMetric(ABC):
    """Base class for all error metrics."""

    @abstractmethod
    def compute(self, array1: Union[NDArray[Any], Sequence[Any]],
                array2: Union[NDArray[Any], Sequence[Any]]):
        """Computes the cost."""


class MeanSquaredError(BaseErrorMetric):
    """Class to compute the error using Mean Squared Error."""

    def compute(self, array1: Union[NDArray[Any], Sequence[Any]],
                array2: Union[NDArray[Any], Sequence[Any]]):
        return mean_squared_error(array1, array2)


class RootMeanSquaredError(BaseErrorMetric):
    """Class to compute the error using Root Mean Squared Error."""

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


class MeanSquaredLogError(BaseErrorMetric):
    """Class to compute the error using the Mean Squared Log Error."""

    def compute(self, array1: Union[NDArray[Any], Sequence[Any]],
                array2: Union[NDArray[Any], Sequence[Any]]):
        return mean_squared_log_error(array1, array2)


class MedianAbsoluteError(BaseErrorMetric):
    """Class to compute the error using the Median Absolute Error."""

    def compute(self, array1: Union[NDArray[Any], Sequence[Any]],
                array2: Union[NDArray[Any], Sequence[Any]]):
        return median_absolute_error(array1, array2)


class MeanAbsolutePercentageError(BaseErrorMetric):
    """Class to compute the error using the Mean Absolute Percentage Error."""

    def compute(self, array1: Union[NDArray[Any], Sequence[Any]],
                array2: Union[NDArray[Any], Sequence[Any]]):
        return mean_absolute_percentage_error(array1, array2)


class Errors:
    """Class to enumerate all available error metrics."""

    MSE = MeanSquaredError()
    RMSE = RootMeanSquaredError()
    MAE = MeanAbsoluteError()
    MAX = MaxError()
    MSLE = MeanSquaredLogError()
    MdAE = MedianAbsoluteError()
    MAPE = MeanAbsolutePercentageError()


def compute_all_errors(array1: Union[NDArray[Any], Sequence[Any]],
                       array2: Union[NDArray[Any], Sequence[Any]]) -> List[Tuple[str, float]]:
    """Function to compute all available errors."""

    computed_errors = list()

    for name, err_comp in Errors.__dict__.items():
        if name.startswith('__'):
            continue
        try:
            computed_errors.append((name, err_comp.compute(array1, array2)))
        except ValueError:
            pass

    return computed_errors
