from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BaseDataGenerator(ABC):
    """Base class to generate data points."""

    def __init__(self) -> None:
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
    def generate(self, no_of_points: Optional[int] = 1):
        """Generates the respective data points."""


class LinearDataGenerator(BaseDataGenerator, ABC):
    """Base class to generate linear data points."""

    @abstractmethod
    def generate(self, no_of_points: Optional[int] = 10, is_increasing: Optional[bool] = False):
        """Generates the respective data points."""


class FloatingPointGenerator(BaseDataGenerator):

    def generate(self, no_of_points: Optional[int] = 1) -> float:
        """Generates a single random float value."""

        return np.random.uniform()


class Normal1DGenerator(BaseDataGenerator):

    def generate(self, no_of_points: Optional[int] = 1):
        """Generates an array of random float values."""

        return np.random.standard_normal(no_of_points)


class Normal2DGenerator(BaseDataGenerator):

    def generate(self, no_of_points: Optional[int] = 1):
        """Generates a 2D array of random float values."""

        return np.random.standard_normal(size=(no_of_points, 2))


class Normal3DGenerator(BaseDataGenerator):

    def generate(self, no_of_points: Optional[int] = 1):
        """Generates a 3D array of random float values."""

        return np.random.standard_normal(size=(no_of_points, 3))


class Linear1DGenerator(LinearDataGenerator):

    def generate(self, no_of_points: Optional[int] = 10, is_increasing: Optional[bool] = False):
        """Generates a 1D array of random float values in a linearly ascending or descending order."""

        if not is_increasing:
            return np.array([i + (np.random.uniform(1.5, 3) * np.random.standard_normal())
                             for i in reversed(range(no_of_points))])

        return np.array([i + (np.random.uniform(1.5, 3) * np.random.standard_normal())
                         for i in range(no_of_points)])

