from abc import ABC, abstractmethod

import numpy as np
from sklearn.datasets import make_classification


class BaseDataGenerator(ABC):
    """Base class to generate data points."""

    def __init__(self, random: bool = False, random_state: int = -1) -> None:

        if random:
            np.random.seed(seed=None)
            return

        self._seed: int = 0 if random_state == -1 else random_state
        np.random.seed(seed=self._seed)

    @property
    def seed_value(self) -> int:
        """Shows the value of the seed."""

        return self._seed

    def set_seed(self, new_seed: int) -> None:
        """
        Sets a new value for the seed.

        :param new_seed: Integer value to set as the seed.
        """

        self._seed = new_seed
        np.random.seed(seed=self._seed)

    @abstractmethod
    def generate(self, no_of_points: int = 1):
        """Generates the respective data points."""


class NormalDataGenerator(BaseDataGenerator):
    """Base class to generate normal data points."""

    @abstractmethod
    def generate(self, no_of_points: int = 1):
        """Generates the respective normal data points."""


class LinearDataGenerator(BaseDataGenerator):
    """Base class to generate linear data points."""

    @abstractmethod
    def generate(self, no_of_points: int = 10, is_increasing: bool = False):
        """Generates the respective linear data points."""


class ClassificationDataGenerator(BaseDataGenerator):
    """Base class to generate data for classification."""

    @abstractmethod
    def generate(self, no_of_points: int = 10, no_classes: int = 2):
        """Generates the respective data points for classification."""


class FloatingPointGenerator(NormalDataGenerator):

    def generate(self, no_of_points: int = 1) -> float:
        """Generates a single random float value."""

        return np.random.uniform()


class Normal1DGenerator(NormalDataGenerator):

    def generate(self, no_of_points: int = 1):
        """Generates an array of random float values."""

        return np.random.standard_normal(size=(no_of_points, 1))


class Normal2DGenerator(NormalDataGenerator):

    def generate(self, no_of_points: int = 1):
        """Generates a 2D array of random float values."""

        return np.random.standard_normal(size=(no_of_points, 2))


class Normal3DGenerator(NormalDataGenerator):

    def generate(self, no_of_points: int = 1):
        """Generates a 3D array of random float values."""

        return np.random.standard_normal(size=(no_of_points, 3))


class Linear1DGenerator(LinearDataGenerator):

    def generate(self, no_of_points: int = 10, is_increasing: bool = True):
        """Generates a 1D array of random float values in a linearly ascending or descending order."""

        if no_of_points == 0:
            return np.array([np.array([])]).transpose()

        if not is_increasing:
            return np.array([np.array([i + (np.random.uniform(1.75, 7) * np.random.standard_normal())])
                             for i in reversed(range(no_of_points))])

        return np.array([np.array([i + (np.random.uniform(1.75, 7) * np.random.standard_normal())])
                         for i in range(no_of_points)])


class Linear2DGenerator(LinearDataGenerator):

    def generate(self, no_of_points: int = 10, is_increasing: bool = True):
        """Generates a 2D array of random float values in a linearly ascending or descending order."""

        if no_of_points == 0:
            return np.array([[], []]).transpose()

        if not is_increasing:
            return np.array([np.array([i + (np.random.uniform(1.75, 7) * np.random.standard_normal()),
                                       j + (np.random.uniform(1.75, 7) * np.random.standard_normal())])
                             for i, j in list(zip(range(no_of_points), reversed(range(no_of_points))))])

        return np.array([np.array([i + (np.random.uniform(1.75, 7) * np.random.standard_normal()),
                                   i + (np.random.uniform(1.75, 7) * np.random.standard_normal())])
                         for i in range(no_of_points)])


class Linear3DGenerator(LinearDataGenerator):

    def generate(self, no_of_points: int = 10, is_increasing: bool = True):
        """Generates a 3D array of random float values in a linearly ascending or descending order."""

        if no_of_points == 0:
            return np.array([[], [], []]).transpose()

        if not is_increasing:
            return np.array([np.array([i + (np.random.uniform(1.75, 7) * np.random.standard_normal()),
                                       i + (np.random.uniform(1.75, 7) * np.random.standard_normal()),
                                       j + (np.random.uniform(1.75, 7) * np.random.standard_normal())])
                             for i, j in list(zip(range(no_of_points), reversed(range(no_of_points))))])

        return np.array([np.array([i + (np.random.uniform(1.75, 7) * np.random.standard_normal()),
                                   i + (np.random.uniform(1.75, 7) * np.random.standard_normal()),
                                   i + (np.random.uniform(1.75, 7) * np.random.standard_normal())])
                         for i in range(no_of_points)])


class LinearlySeparable2DGenerator(ClassificationDataGenerator):

    def generate(self, no_of_points: int = 10, no_classes: int = 2):
        """Generates linearly separable 2D data along with labels for classification."""

        if no_of_points == 0:
            return np.array([[], [], []]).transpose()

        try:
            random_state = self.seed_value
        except AttributeError:
            random_state = None

        x, y = make_classification(n_samples=no_of_points, n_features=2, n_redundant=0, n_informative=2,
                                   random_state=random_state, n_classes=no_classes, n_clusters_per_class=1)

        return np.array(list(zip(*(x[:, 0], x[:, 1], y))))


class LinearlySeparable3DGenerator(ClassificationDataGenerator):

    def generate(self, no_of_points: int = 10, no_classes: int = 2):
        """Generates linearly separable 3D data along with labels for classification."""

        if no_of_points == 0:
            return np.array([[], [], [], []]).transpose()

        try:
            random_state = self.seed_value
        except AttributeError:
            random_state = None

        x, y = make_classification(n_samples=no_of_points, n_features=3, n_redundant=0, n_informative=3,
                                   random_state=random_state, n_classes=no_classes, n_clusters_per_class=1)

        return np.array(list(zip(*(x[:, 0], x[:, 1], x[:, 2], y))))
