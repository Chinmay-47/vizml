from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Base model for all other ML models."""

    @abstractmethod
    def train(self):
        """Function to train the Model."""

    @abstractmethod
    def predict(self):
        """Function to predict the output of the Model."""
