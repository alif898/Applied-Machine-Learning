from __future__ import annotations
from tqdm import tqdm
from collections.abc import Iterator
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from activation import Activation, ReLU, LeakyReLU, TanH, Softmax


class Layer:
    """
    Single layer within the neural network, equivalent to Keras dense layer

    Can be used as hidden layer (Linear transformation + Activation) and output layer

    All model parameters stored in MLP class
    """
    activation_fn = {
        'leaky_relu': LeakyReLU(),
        'relu': ReLU(),
        'softmax': Softmax(),
        'tanh': TanH()
    }

    def __init__(self, n_units: int, activation: str = 'relu') -> None:
        """
        Constructor for neural network layer

        :param n_units: Number of neurons in the layer
        :param activation: Activation function for the layer, default is ReLU
        """
        self.activation = activation
        self.n_units = n_units
        return

    def get_n_units(self) -> int:
        """
        Getter to return number of neurons in the layer

        :return: Number of neurons in the layer
        """
        return self.n_units

    def get_activation(self) -> Activation:
        """
        Getter to return activation function, using class variable to map from string representation

        :return: Instance of activation function
        """
        return self.activation_fn[self.activation]

    @staticmethod
    def forward_pass(inputs: np.ndarray,
                     weights: np.ndarray,
                     bias: np.ndarray,
                     activation: Activation
                     ) -> tuple[np.ndarray, np.ndarray]:
        """
        Method to perform forward pass based on activation function

        :param inputs: X
        :param weights: Current weights
        :param bias: Current bias
        :param activation: Activation function
        :return: Tuple of (Y post-activation, Y pre-activation)
        """
        z_curr = np.dot(inputs, weights.transpose()) + bias
        a_curr = activation.apply_function(z_curr)

        return a_curr, z_curr

    @staticmethod
    def backpropagation(d_a_curr: np.ndarray,
                        w_curr: np.ndarray,
                        z_curr: np.ndarray,
                        a_prev: np.ndarray,
                        activation: Activation
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Method to perform backpropagation based on activation function

        :param d_a_curr: dA
        :param w_curr: W
        :param z_curr: z
        :param a_prev: A from n-1 layer
        :param activation: Activation function
        :return: Tuple of (Next dA, dW, dB)
        """
        if isinstance(activation, Softmax):
            d_w = np.dot(a_prev.transpose(), d_a_curr)
            d_b = np.sum(d_a_curr, axis=0, keepdims=True)
            d_a = np.dot(d_a_curr, w_curr)
        else:
            d_z = activation.calculate_derivative(d_a_curr, z_curr)
            d_w = np.dot(a_prev.transpose(), d_z)
            d_b = np.sum(d_z, axis=0, keepdims=True)
            d_a = np.dot(d_z, w_curr)

        return d_a, d_w, d_b


class MLP:
    """
    Multilayer Perceptron class

    Note: Y should not be one-hot encoded
    """
    def __init__(self,
                 *args: Layer,
                 learning_rate: float = 0.01
                 ) -> None:
        """
        Constructor for MLP, taking sequence of Layers to build network

        :param args: Sequence of Layer
        :param learning_rate: Learning rate to use when updating parameters, default is 0.01
        """
        self.layers = args
        self.learning_rate = learning_rate

        # Initialise variables that will store various parameters of the neural network
        self.architecture: list[dict[str, int | Activation]] = list()
        self.cache: list[dict[str, np.ndarray]] = [{}] * len(args)
        self.gradients: list[dict[str, np.ndarray]] = list()
        self.loss_history: list[float] = list()
        self.parameters: list[dict[str, np.ndarray]] = list()
        return

    @staticmethod
    def cross_entropy(y_pred: np.ndarray, y_test: np.ndarray, epsilon: float = 1e-12) -> float:
        """
        Calculates cross-entropy loss between predicted & actual values of Y

        :param y_pred: Predicted values of Y
        :param y_test: Actual values of Y
        :param epsilon: Value to clip predictions against, to prevent overflow
        :return: Cross-entropy loss
        """

        # Adjusted to ensure numerical stability, prevent overflow
        clipped_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        n_samples = len(y_test)
        log_probs = -np.log(clipped_pred[range(n_samples), y_test])
        loss = np.sum(log_probs) / n_samples

        return loss

    @staticmethod
    def evaluate_accuracy(y_pred: np.ndarray, y_test: np.ndarray) -> float:
        """
        Evaluates performance of trained model

        :param y_pred: Predicted values of Y
        :param y_test: Actual values of Y
        :return: Accuracy of model
        """
        return sum(y_pred == y_test) / y_test.shape[0]

    @staticmethod
    def check_gradients(mlp: MLP, x: np.ndarray, y: np.ndarray) -> float:
        """
        Static method to verify gradients

        :param mlp: Model to verify gradients for
        :param x: X
        :param y: Y
        :return: Finite difference approximation of gradient
        """

        # Generate target labels
        n_classes = len(np.unique(y))
        labels = np.eye(n_classes)[np.array(y, dtype=np.int16)]

        predictions = mlp.perform_forward_pass(x)

        return scipy.optimize.check_grad(predictions, mlp.perform_backpropagation(predictions, labels), x)

    @staticmethod
    def generate_minibatch(x: np.ndarray,
                           y: np.ndarray,
                           batch_size: int,
                           shuffle: bool = False
                           ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Static method that generates the batches

        :param x: X
        :param y: Y
        :param batch_size: Size of each minibatch
        :param shuffle: Whether to shuffle the training data when generating the batches
        :return: Iterator that generates the next set of X and Y minibatch
        """
        indexes = np.arange(x.shape[0])
        if shuffle:
            np.random.shuffle(indexes)
        for start in range(0, x.shape[0] - batch_size + 1, batch_size):
            end = start + batch_size
            if shuffle:
                next_indexes = indexes[start: end]
            else:
                next_indexes = slice(start, end)
            yield x[next_indexes], y[next_indexes]

    def compile(self, x: np.ndarray) -> MLP:
        """
        Constructs neural network architecture based on training data

        Maps input dimensions to output dimensions across the layers

        :param x: Training set of X
        :return: Model with architecture
        """
        d = x.shape[1]

        for idx, layer in enumerate(self.layers):
            if idx == 0:
                # Building first layer
                self.architecture.append(
                    {
                        'input_dim': d,
                        'output_dim': self.layers[idx].get_n_units(),
                        'activation': self.layers[idx].get_activation()
                    }
                )
            elif 0 < idx < len(self.layers) - 1:
                # Building intermediate hidden layers
                self.architecture.append(
                    {
                        'input_dim': self.layers[idx - 1].get_n_units(),
                        'output_dim': self.layers[idx].get_n_units(),
                        'activation': self.layers[idx].get_activation()
                    }
                )
            else:
                # Building output layer
                self.architecture.append(
                    {
                        'input_dim': self.layers[idx - 1].get_n_units(),
                        'output_dim': self.layers[idx].get_n_units(),
                        'activation': Softmax()
                    }
                )

        # Handling edge case of no hidden layers
        if len(self.layers) == 1:
            self.architecture[-1]['activation'] = Softmax()

        return self

    def init_weights(self, x: np.ndarray) -> MLP:
        """
        Initialise starting weights of the model

        :param x: Training set of X
        :return: Model with weights initialised
        """

        # Compiles network architecture
        self.compile(x)

        # Set random seed for reproducibility
        np.random.seed(0)

        for i in range(len(self.architecture)):
            self.parameters.append(
                {
                    'w': np.random.uniform(
                        low=-1,
                        high=1,
                        size=(
                            self.architecture[i]['output_dim'],
                            self.architecture[i]['input_dim']
                        )
                    ),
                    'b': np.zeros((
                        1, self.architecture[i]['output_dim']
                    ))
                }
            )

        return self

    def perform_forward_pass(self, x: np.ndarray) -> np.ndarray:
        """
        Iterates through layers and performs one full forward pass through network

        :param x: X
        :return: Predicted values given current parameters
        """
        a_curr = x

        for i in range(len(self.parameters)):
            a_prev = a_curr
            a_curr, z_curr = self.layers[i].forward_pass(
                inputs=a_prev,
                weights=self.parameters[i]['w'],
                bias=self.parameters[i]['b'],
                activation=self.architecture[i]['activation']
            )

            self.cache[i] = {
                'inputs': a_prev,
                'z': z_curr
            }

        return a_curr

    def perform_backpropagation(self, predicted: np.ndarray, actual: np.ndarray) -> None:
        """
        Iterates through layers in reverse order and performs backpropagation through network

        :param predicted: Predicted values of Y
        :param actual: Actual values of Y
        :return: None
        """
        n_samples = len(actual)

        # Compute gradients on predictions
        d_scores = predicted
        d_scores[range(n_samples), actual] -= 1
        d_scores /= n_samples

        d_a_prev = d_scores
        for idx, layer in reversed(list(enumerate(self.layers))):
            d_a_curr = d_a_prev

            a_prev = self.cache[idx]['inputs']
            z_curr = self.cache[idx]['z']
            w_curr = self.parameters[idx]['w']

            activation = self.architecture[idx]['activation']

            d_a_prev, d_w_curr, d_b_curr = layer.backpropagation(
                d_a_curr=d_a_curr,
                w_curr=w_curr,
                z_curr=z_curr,
                a_prev=a_prev,
                activation=activation
            )

            self.gradients.append(
                {
                    'd_w': d_w_curr,
                    'd_b': d_b_curr
                }
            )

    def update_parameters(self) -> None:
        """
        Method that will iterate through layers and update parameters after forward pass & backpropagation

        :return: None
        """
        for idx, layer in enumerate(self.layers):
            self.parameters[idx]['w'] -= self.learning_rate * list(reversed(self.gradients))[idx]['d_w'].transpose()
            self.parameters[idx]['b'] -= self.learning_rate * list(reversed(self.gradients))[idx]['d_b']

    def fit(self,
            x: np.ndarray,
            y: np.ndarray,
            epochs: int = 20,
            use_minibatch: bool = True,
            batch_size: int = 64
            ) -> MLP:
        """
        Takes in training set of X, Y to learn parameters

        Note: Y should not be one-hot encoded

        :param x: Training set of X
        :param y: Training set of Y
        :param epochs: Number of epochs to run, default set as 20
        :param use_minibatch: Whether to use minibatch
        :param batch_size: Size of each minibatch, default size is 64
        :return: Trained model with learned parameters
        """

        # Initialise weights of model
        self.init_weights(x)

        for _ in tqdm(range(epochs)):
            if not use_minibatch:
                # Perform both forward pass & backpropagation
                y_pred = self.perform_forward_pass(x)
                self.perform_backpropagation(y_pred, y)

                # Perform updating of layer weights
                self.update_parameters()

                # Record loss
                loss = MLP.cross_entropy(y_pred, y)
                self.loss_history.append(loss)

            if use_minibatch:
                for batch in MLP.generate_minibatch(x, y, batch_size=batch_size, shuffle=True):
                    next_batch_x, next_batch_y = batch

                    # Perform both forward pass & backpropagation
                    y_pred = self.perform_forward_pass(next_batch_x)
                    self.perform_backpropagation(y_pred, next_batch_y)

                    # Perform updating of layer weights
                    self.update_parameters()

                # Record loss
                loss = MLP.cross_entropy(self.perform_forward_pass(x), y)
                self.loss_history.append(loss)

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts values of Y given current parameters

        :param x: Test set of X
        :return: Predicted Y
        """
        return self.perform_forward_pass(x).argmax(axis=1)

    def plot_loss(self, plot_title: str = None) -> None:
        """
        Plots cross-entropy loss against number of epochs

        :param plot_title: Title to use for the plot
        :return: Cross-entropy loss against epochs
        """

        # Prints model architecture
        print('-------')
        for idx, layer in enumerate(self.architecture):
            print(f'Layer {idx}')
            for k, v in layer.items():
                print(f'{k}: {v}')
            print('-------')

        plt.clf()
        fig, ax = plt.subplots()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax.plot([i for i in range(len(self.loss_history))], self.loss_history)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('CE Loss')

        if plot_title:
            ax.set_title(plot_title)
        else:
            ax.set_title('CE Loss against Epoch')
        return
