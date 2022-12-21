from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Callable
import numpy as np

from multiclassregression import MulticlassRegression


class BaseGradientDescent(ABC):
    """
    Abstract class to represent various gradient descent methods
    """
    @abstractmethod
    def __init__(self, momentum: float = 0.0, schedule: Callable[[int], float] = None) -> None:
        """
        Abstract constructor for any gradient descent method
        Optional usage of momentum, Robbins-Monro schedule available
        :param momentum: Value of momentum
        :param schedule: Function that generates next learning rate given iteration t
        """
        self.momentum = momentum
        self.schedule = schedule
        return

    @abstractmethod
    def pass_params(self) -> dict:
        """
        Abstract method to pass hyperparameters to driver function in MulticlassRegression
        :return: Dictionary containing hyperparameter name as key, value as value
        """
        pass


class StochasticGradientDescent(BaseGradientDescent):
    """
    Stochastic Gradient Descent
    """

    def __init__(self, momentum: float = 0.0, schedule: Callable[[int], float] = None) -> None:
        super().__init__(momentum=momentum, schedule=schedule)
        return

    def pass_params(self) -> dict:
        return {
            'function': MulticlassRegression.run_sgd,
            'momentum': self.momentum,
            'schedule': self.schedule
        }


class MinibatchStochasticGradientDescent(StochasticGradientDescent):
    """
    Stochastic Gradient Descent with Minibatch
    Inherits from StochasticGradientDescent
    """

    def __init__(self,
                 batch_size: int = 16,
                 momentum: float = 0.0,
                 schedule: Callable[[int], float] = None,
                 shuffle: bool = False
                 ) -> None:
        """
        2 additional optional parameters, batch_size & shuffle
        :param batch_size: Size of each minibatch
        :param shuffle: Whether to shuffle the training data when generating the batches
        """
        super().__init__(momentum=momentum, schedule=schedule)
        self.batch_size = batch_size
        self.shuffle = shuffle
        return

    @staticmethod
    def iterate_minibatch(x: np.ndarray,
                          y: np.ndarray,
                          batch_size: int,
                          shuffle: bool
                          ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Static method that generates the batches
        :param x: Training set of X
        :param y: Training set of Y
        :param batch_size: Size of each minibatch
        :param shuffle: Whether to shuffle the training data when generating the batches
        :return: Iterator that generates the next set of X and Y minibatch
        """
        indexes = np.arange(x.shape[0])
        if shuffle:
            np.random.shuffle(indexes)

        # Iterate through indexes in batches based on batch size
        for start in range(0, x.shape[0] - batch_size + 1, batch_size):
            end = start + batch_size
            if shuffle:
                next_indexes = indexes[start: end]
            else:
                next_indexes = slice(start, end)
            yield x[next_indexes], y[next_indexes]

    def pass_params(self) -> dict:
        params = super().pass_params()
        params['function'] = MulticlassRegression.run_minibatch_sgd
        params['batch_size'] = self.batch_size
        params['shuffle'] = self.shuffle
        return params
