from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from sklearn.datasets import make_classification, make_moons, make_circles


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
    def generate(self, no_of_points: int = 10):
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

    def generate(self, no_of_points: int = 10):
        """Generates linearly separable 2D data along with labels for classification."""

        if no_of_points == 0:
            return np.array([[], [], []]).transpose()

        random_state: Optional[int]
        try:
            random_state = self.seed_value
        except AttributeError:
            random_state = None

        x, y = make_classification(n_samples=no_of_points, n_features=2, n_redundant=0, n_informative=2,
                                   random_state=random_state, n_clusters_per_class=1)

        x += 1.5 * np.random.uniform(size=x.shape)

        return np.concatenate((x, np.expand_dims(y, axis=1)), axis=1)


class LinearlySeparable3DGenerator(ClassificationDataGenerator):

    def generate(self, no_of_points: int = 10):
        """Generates linearly separable 3D data along with labels for classification."""

        if no_of_points == 0:
            return np.array([[], [], [], []]).transpose()

        random_state: Optional[int]
        try:
            random_state = self.seed_value
        except AttributeError:
            random_state = None

        x, y = make_classification(n_samples=no_of_points, n_features=3, n_redundant=0, n_informative=3,
                                   random_state=random_state, n_clusters_per_class=1)

        x += 1.5 * np.random.uniform(size=x.shape)

        return np.concatenate((x, np.expand_dims(y, axis=1)), axis=1)


class MoonData2DGenerator(ClassificationDataGenerator):

    def generate(self, no_of_points: int = 10):
        """Generates moon shaped 2D data along with labels for classification."""

        if no_of_points == 0:
            return np.array([[], [], []]).transpose()

        random_state: Optional[int]
        try:
            random_state = self.seed_value
        except AttributeError:
            random_state = None

        x, y = make_moons(n_samples=no_of_points, noise=0.2, random_state=random_state)

        return np.concatenate((x, np.expand_dims(y, axis=1)), axis=1)


class MoonData3DGenerator(ClassificationDataGenerator):

    def generate(self, no_of_points: int = 10):
        """Generates moon shaped 3D data along with labels for classification."""

        if no_of_points == 0:
            return np.array([[], [], [], []]).transpose()

        random_state: Optional[int]
        try:
            random_state = self.seed_value
        except AttributeError:
            random_state = None

        x, y = make_moons(n_samples=no_of_points, noise=0.2, random_state=random_state)

        x = np.array([np.concatenate([item, np.array([1.5 * np.random.random()])]) for item in x])

        return np.concatenate((x, np.expand_dims(y, axis=1)), axis=1)


class CircleDataGenerator(ClassificationDataGenerator):

    def generate(self, no_of_points: int = 10):
        """Generates circular 2D data along with labels for classification."""

        if no_of_points == 0:
            return np.array([[], [], []]).transpose()

        random_state: Optional[int]
        try:
            random_state = self.seed_value
        except AttributeError:
            random_state = None

        x, y = make_circles(n_samples=no_of_points, noise=0.2, factor=0.5, random_state=random_state)

        return np.concatenate((x, np.expand_dims(y, axis=1)), axis=1)


class SphericalDataGenerator(ClassificationDataGenerator):

    def generate(self, no_of_points: int = 10):
        """Generates spherical 3D data along with labels for classification."""

        if no_of_points == 0:
            return np.array([[], [], [], []]).transpose()

        random_state: Optional[int]
        try:
            random_state = self.seed_value
        except AttributeError:
            random_state = None

        x, y = make_circles(n_samples=no_of_points, noise=0.2, factor=0.5, random_state=random_state)

        x = np.array([np.concatenate([item, np.array([1.5 * np.random.random()])]) for item in x])

        return np.concatenate((x, np.expand_dims(y, axis=1)), axis=1)
