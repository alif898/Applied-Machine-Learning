from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class Activation(ABC):
    """
    Activation function within a neural network layer
    """
    @abstractmethod
    def apply_function(self, x: np.ndarray) -> np.ndarray:
        """
        Method to apply activation function

        :param x: X
        :return: Result after passing X through activation function
        """
        pass

    @abstractmethod
    def calculate_derivative(self, d_a: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Method to calculate gradient for backpropagation

        :param d_a: dA
        :param z: z
        :return: Derivative
        """
        pass

    def __str__(self) -> str:
        """
        Prints class name

        :return: Class name
        """
        return str(type(self))[17:-2]


class LeakyReLU(Activation):

    def __init__(self, alpha: float = 0.01) -> None:
        """
        Leaky Rectified Linear Unit activation function

        :param alpha: Default is 0.01
        """
        self.alpha = alpha

    def apply_function(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.alpha * x)

    def calculate_derivative(self, d_a: np.ndarray, z: np.ndarray) -> np.ndarray:
        d_z = np.array(d_a, copy=True)
        d_z[z < 0] = self.alpha
        return d_z


class ReLU(LeakyReLU):

    def __init__(self) -> None:
        """
        Special case of LeakyReLU where alpha is 0
        """
        super().__init__(alpha=0.0)

    def calculate_derivative(self, d_a: np.ndarray, z: np.ndarray) -> np.ndarray:
        d_z = np.array(d_a, copy=True)
        d_z[z <= 0] = 0
        return d_z


class TanH(Activation):

    def apply_function(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def calculate_derivative(self, d_a: np.ndarray, z: np.ndarray) -> np.ndarray:
        d_z = np.array(d_a, copy=True)
        return 1 - (self.apply_function(d_z) ** 2)


class Softmax(Activation):

    def apply_function(self, x: np.ndarray) -> np.ndarray:

        # Adjusted to ensure numerical stability, prevent overflow
        s = np.max(x, axis=1)
        s = s[:, np.newaxis]
        exp_x = np.exp(x - s)
        div = np.sum(exp_x, axis=1)
        div = div[:, np.newaxis]

        return exp_x / div

    def calculate_derivative(self, d_a: np.ndarray, z: np.ndarray) -> np.ndarray:
        # Used exclusively as output layer
        raise NotImplementedError
