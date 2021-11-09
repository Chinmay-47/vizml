from abc import ABC, abstractmethod

import numpy as np


class BaseDataGenerator(ABC):
    """Base class to generate data points."""

    def __init__(self, random: bool = False, random_state: int = -1) -> None:

        if random:
            np.random.seed(seed=None)
            return
        if random_state != -1:
            self._seed: int = random_state
            np.random.seed(seed=self._seed)
            return
        self._seed: int = 0
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
