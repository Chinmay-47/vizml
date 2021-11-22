from abc import ABC, abstractmethod
from sklearn.metrics import silhouette_samples, silhouette_score
from numpy.typing import NDArray
from typing import Sequence, Union, Any


class BaseErrorMetric(ABC):
    """Base class for all error metrics."""

    @abstractmethod
    def compute(self, array1: Union[NDArray[Any], Sequence[Any]],
                array2: Union[NDArray[Any], Sequence[Any]]):
        """Computes the metric."""


class AvgSilhouetteScore(BaseErrorMetric):
    """
    Class to compute the Mean Silhouette Coefficient of all samples.
    """

    def compute(self, array1: Union[NDArray[Any], Sequence[Any]],
                array2: Union[NDArray[Any], Sequence[Any]]):
        """Computes the metric"""

        return silhouette_score(array1, array2)


class AllSilhouetteScores(BaseErrorMetric):
    """
    Class to compute the Silhouette Coefficient for each sample.
    """

    def compute(self, array1: Union[NDArray[Any], Sequence[Any]],
                array2: Union[NDArray[Any], Sequence[Any]]):
        """Computes the metric"""

        return silhouette_samples(array1, array2)
