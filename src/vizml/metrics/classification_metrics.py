from abc import ABC, abstractmethod
from enum import Enum
from typing import Sequence, Union, Any, List, Tuple

from numpy.typing import NDArray
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, log_loss, roc_auc_score,
                             hinge_loss, cohen_kappa_score, hamming_loss, jaccard_score, matthews_corrcoef,
                             balanced_accuracy_score, confusion_matrix)


class BaseErrorMetric(ABC):
    """Base class for all error metrics."""

    @abstractmethod
    def compute(self, array1: Union[NDArray[Any], Sequence[Any]],
                array2: Union[NDArray[Any], Sequence[Any]]):
        """Computes the cost."""


class Accuracy(BaseErrorMetric):
    """Class to compute the Accuracy Score."""

    def compute(self, array1: Union[NDArray[Any], Sequence[Any]],
                array2: Union[NDArray[Any], Sequence[Any]]):
        return accuracy_score(array1, array2)


class BalancedAccuracy(BaseErrorMetric):
    """Class to compute the Balanced Accuracy Score."""

    def compute(self, array1: Union[NDArray[Any], Sequence[Any]],
                array2: Union[NDArray[Any], Sequence[Any]]):
        return balanced_accuracy_score(array1, array2)


class F1Score(BaseErrorMetric):
    """Class to compute the F1 Score."""

    def compute(self, array1: Union[NDArray[Any], Sequence[Any]],
                array2: Union[NDArray[Any], Sequence[Any]]):
        return f1_score(array1, array2)


class Precision(BaseErrorMetric):
    """Class to compute the Precision Score."""

    def compute(self, array1: Union[NDArray[Any], Sequence[Any]],
                array2: Union[NDArray[Any], Sequence[Any]]):
        return precision_score(array1, array2)


class Recall(BaseErrorMetric):
    """Class to compute the Recall Score."""

    def compute(self, array1: Union[NDArray[Any], Sequence[Any]],
                array2: Union[NDArray[Any], Sequence[Any]]):
        return recall_score(array1, array2)


class CohenKappaScore(BaseErrorMetric):
    """Class to compute the Cohen Kappa Score."""

    def compute(self, array1: Union[NDArray[Any], Sequence[Any]],
                array2: Union[NDArray[Any], Sequence[Any]]):
        return cohen_kappa_score(array1, array2)


class HammingLoss(BaseErrorMetric):
    """Class to compute the Hamming Loss."""

    def compute(self, array1: Union[NDArray[Any], Sequence[Any]],
                array2: Union[NDArray[Any], Sequence[Any]]):
        return hamming_loss(array1, array2)


class JaccardSimilarity(BaseErrorMetric):
    """Class to compute the Jaccard Similarity Score."""

    def compute(self, array1: Union[NDArray[Any], Sequence[Any]],
                array2: Union[NDArray[Any], Sequence[Any]]):
        return jaccard_score(array1, array2)


class MatthewsCorrelation(BaseErrorMetric):
    """Class to compute the Matthews Correlation Coefficient."""

    def compute(self, array1: Union[NDArray[Any], Sequence[Any]],
                array2: Union[NDArray[Any], Sequence[Any]]):
        return matthews_corrcoef(array1, array2)


class LogLoss(BaseErrorMetric):
    """Class to compute the Log Loss."""

    def compute(self, array1: Union[NDArray[Any], Sequence[Any]],
                array2: Union[NDArray[Any], Sequence[Any]]):
        return log_loss(array1, array2)


class HingeLoss(BaseErrorMetric):
    """Class to compute the Hinge Loss."""

    def compute(self, array1: Union[NDArray[Any], Sequence[Any]],
                array2: Union[NDArray[Any], Sequence[Any]]):
        return hinge_loss(array1, array2)


class RocAucScore(BaseErrorMetric):
    """Class to compute the Hinge Loss."""

    def compute(self, array1: Union[NDArray[Any], Sequence[Any]],
                array2: Union[NDArray[Any], Sequence[Any]]):
        return roc_auc_score(array1, array2)


class ConfusionMatrix(BaseErrorMetric):
    """Class to compute the Confusion Matrix"""

    def compute(self, array1: Union[NDArray[Any], Sequence[Any]],
                array2: Union[NDArray[Any], Sequence[Any]]):
        return confusion_matrix(array1, array2)


class Metric(Enum):
    """Class to enumerate all available metrics."""

    ACC = Accuracy()
    BAL_ACC = BalancedAccuracy()
    F1 = F1Score()
    PRECISION = Precision()
    RECALL = Recall()
    KAPPA = CohenKappaScore()
    HAMMING = HammingLoss()
    JACCARD = JaccardSimilarity()
    MCC = MatthewsCorrelation()


class ProbMetric(Enum):
    """Class to enumerate available metrics that use decision probabilities."""

    LOG = LogLoss()
    HINGE = HingeLoss()
    ROC_AUC = RocAucScore()


def compute_all_metrics(array1: Union[NDArray[Any], Sequence[Any]],
                        array2: Union[NDArray[Any], Sequence[Any]],
                        rounding: int = 3) -> List[Tuple[str, float]]:
    """Function to compute all available metrics."""

    computed_metrics = list()

    for name, metric in Metric.__members__.items():
        computed_metrics.append((name, round(metric.value.compute(array1, array2), rounding)))

    return computed_metrics


def compute_all_prob_metrics(array1: Union[NDArray[Any], Sequence[Any]],
                             array2: Union[NDArray[Any], Sequence[Any]],
                             rounding: int = 3) -> List[Tuple[str, float]]:
    """Function to compute all available metrics that use decision probabilities."""

    computed_metrics = list()

    for name, metric in ProbMetric.__members__.items():
        computed_metrics.append((name, round(metric.value.compute(array1, array2), rounding)))

    return computed_metrics
