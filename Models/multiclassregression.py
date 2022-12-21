from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from gradientdescent import BaseGradientDescent, StochasticGradientDescent, MinibatchStochasticGradientDescent


class MulticlassRegression:
    """
    MulticlassRegression classifier that implements gradient descent
    """

    def __init__(self,
                 epsilon: float = 1e-8,
                 gradient_descent: BaseGradientDescent = StochasticGradientDescent(),
                 learning_rate: float = 0.1,
                 max_iters: float = 1e4,
                 n_classes: int = None,
                 ) -> None:
        """
        Initialises MulticlassRegression with choice of learning rate and max iterations
        :param epsilon: Threshold to check change in gradients against
        :param gradient_descent: Gradient descent method to use, default is Stochastic Gradient Descent
        :param n_classes: Default is None, model can infer from dataset during training
        :param learning_rate: Default set as 0.1
        :param max_iters: Default set as 1e4
        """
        self.epsilon = epsilon
        self.gradient_descent = gradient_descent
        self.learning_rate = learning_rate
        self.loss_history = list()
        self.max_iters = max_iters
        self.n_classes = n_classes
        self.weights = None
        return

    @staticmethod
    def process_x(x: np.ndarray) -> tuple[np.ndarray, int]:
        """
        Static method to process X if it is 1D array
        :param x: Training or test set of X
        :return: Tuple of (X as 2D array, n)
        """
        if x.ndim == 1:
            x = x[:, None]
        n = x.shape[0]
        return x, n

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts values of Y given current weights
        Same function used in training & prediction time
        :param x: Training or test set of X
        :return: Predicted Y
        """
        x, n = MulticlassRegression.process_x(x)
        y_pred = np.exp(np.matmul(x, self.weights))
        return y_pred / y_pred.sum(axis=1).reshape(n, 1)

    def calculate_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate gradients, using difference between predicted Y at current weights and actual Y
        :param x: Training set of X
        :param y: Training set of Y
        :return: Gradients
        """
        return np.matmul(x.transpose(), self.predict(x) - y)

    def cross_entropy(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculates cross-entropy loss by predicting on X and comparing to Y
        :param x: X
        :param y: Y
        :return: Cross-entropy loss
        """
        return -np.sum(y * np.log(self.predict(x))) / y.shape[0]

    def run_sgd(self, x: np.ndarray, y: np.ndarray, print_loss: bool = False) -> None:
        """
        Driver function that performs Stochastic Gradient Descent
        :param x: Training set of X
        :param y: Training set of Y
        :param print_loss: Whether to print the loss at each iteration
        :return: Weights will be updated
        """
        delta = 0.0

        # Retrieve hyperparameters from gradient descent object
        params = self.gradient_descent.pass_params()
        momentum = params['momentum']
        schedule = params['schedule']

        # Online update of gradients using 1 sample at a time
        for t in range(x.shape[0]):
            gradients = self.calculate_gradient(x[t, None], y[t, None])

            # Calculates change in weights using momentum
            delta = (momentum * delta) + ((1 - momentum) * gradients)
            self.weights = self.weights - (self.learning_rate * delta)

            # Record loss and print loss
            loss = self.cross_entropy(x, y)
            self.loss_history.append(loss)
            if print_loss:
                print(f'Iteration {t}: Training loss is {loss}')

            # Update learning rate
            if schedule:
                try:
                    self.learning_rate = schedule(t)
                except ZeroDivisionError:
                    pass

            # 2 stopping criteria used, when gradient change is small or when maximum iterations reached
            if not (np.linalg.norm(gradients) > self.epsilon and t < self.max_iters):
                break

    def run_minibatch_sgd(self, x: np.ndarray, y: np.ndarray, print_loss: bool = False) -> None:
        """
        Driver function that performs Minibatch Stochastic Gradient Descent
        :param x: Training set of X
        :param y: Training set of Y
        :param print_loss: Whether to print the loss at each iteration
        :return: Weights will be updated
        """
        delta = 0.0

        # Retrieve hyperparameters from gradient descent object
        params = self.gradient_descent.pass_params()
        batch_size = params['batch_size']
        momentum = params['momentum']
        schedule = params['schedule']
        shuffle = params['shuffle']

        # Verify validity of batch size
        if batch_size > x.shape[0]:
            raise ValueError('Batch size larger than number of training samples')

        # Online update of gradients using batches of X and Y
        for t, batch in enumerate(
                MinibatchStochasticGradientDescent.iterate_minibatch(x, y, batch_size, shuffle)
        ):
            next_batch_x, next_batch_y = batch
            gradients = self.calculate_gradient(next_batch_x, next_batch_y)

            # Calculates change in weights using momentum
            delta = (momentum * delta) + ((1 - momentum) * gradients)
            self.weights = self.weights - (self.learning_rate * delta)

            # Record loss and print loss
            loss = self.cross_entropy(x, y)
            self.loss_history.append(loss)
            if print_loss:
                print(f'Iteration {t}: Training loss is {loss}')

            # Update learning rate
            if schedule:
                try:
                    self.learning_rate = schedule(t)
                except ZeroDivisionError:
                    pass

            # 2 stopping criteria used, when gradient change is small or when maximum iterations reached
            if not (np.linalg.norm(gradients) > self.epsilon and t < self.max_iters):
                break

    def fit(self, x: np.ndarray, y: np.ndarray, print_loss: bool = False) -> MulticlassRegression:
        """
        Takes in training X, Y to calculate weights using gradient descent
        :param x: Training set of X
        :param y: Training set of Y
        :param print_loss: Whether to print the loss at each iteration
        :return: Trained model with learned weights
        """

        # If y only has 2 classes, input may be passed in as 1D array
        self.n_classes = y.shape[1] if len(y.shape) > 1 else 1
        x, _ = MulticlassRegression.process_x(x)
        n, d = x.shape

        # Initialise weights
        self.weights = np.random.rand(d, self.n_classes)

        # Update weights through specific gradient descent function
        gradient_descent_fn = self.gradient_descent.pass_params()['function']
        gradient_descent_fn(self, x, y, print_loss=print_loss)

        return self

    def plot_loss(self) -> None:
        """
        :return: Plots cross entropy loss against iterations of gradient descent
        """
        plt.clf()
        fig, ax = plt.subplots()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax.plot([i for i in range(len(self.loss_history))], self.loss_history)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('CE Loss')

        model_name = str(type(self))[17:-2]
        gd_name = str(type(self.gradient_descent))[17:-2]
        ax.set_title(
            f'{model_name} with {gd_name}'
        )
        return

    @staticmethod
    def evaluate_accuracy(y_pred: np.ndarray,
                          y_test: np.ndarray
                          ) -> float:
        """
        Evaluates performance of trained model
        :param y_pred: Predicted values of Y
        :param y_test: Actual values of Y
        :return: Accuracy of model
        """
        return sum(y_pred.argmax(axis=1) == y_test.argmax(axis=1)) / y_test.shape[0]


class LogisticRegression(MulticlassRegression):
    """
    LogisticRegression classifier that implements gradient descent
    Inherits from MulticlassRegression
    """

    def __init__(self,
                 epsilon: float = 1e-8,
                 gradient_descent: BaseGradientDescent = StochasticGradientDescent(),
                 learning_rate: float = 0.1,
                 max_iters: float = 1e4
                 ) -> None:
        """
        Constructor that utilises parent class constructor with n_classes=2
        :param epsilon: Threshold to check change in gradients against
        :param gradient_descent: Gradient descent method to use, default is Stochastic Gradient Descent
        :param learning_rate: Default set as 0.1
        :param max_iters: Default set as 1e4
        """
        super().__init__(n_classes=2,
                         epsilon=epsilon,
                         gradient_descent=gradient_descent,
                         learning_rate=learning_rate,
                         max_iters=max_iters)
        return

    def predict(self, x: np.ndarray, is_training: bool = False) -> np.ndarray:
        """
        Override parent class method and use Logistic function instead
        Predicts values of Y given current weights
        Same function used in training & prediction time
        :param x: Training or test set of X
        :param is_training: Whether method is called in training or prediction
        :return: Predicted Y
        """
        x, n = super().process_x(x)

        def logistic(z: np.ndarray) -> np.ndarray:
            return 1.0 / (1 + np.exp(-z))

        y_pred = logistic(np.dot(x, self.weights))
        if is_training:
            return y_pred
        else:
            return y_pred.transpose()[0]

    def calculate_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Override parent class method and divides by n
        Calculate gradients, using difference between predicted Y at current weights and actual Y
        :param x: Training set of X
        :param y: Training set of Y
        :return: Gradients
        """
        x, n = super().process_x(x)
        return np.dot(x.transpose(), self.predict(x, is_training=True) - y) / n

    @staticmethod
    def evaluate_accuracy(y_pred: np.ndarray,
                          y_test: np.ndarray
                          ) -> float:
        """
        Evaluates performance of trained model
        :param y_pred: Predicted values of Y
        :param y_test: Actual values of Y
        :return: Accuracy of model
        """
        return sum(y_pred == y_test) / y_test.shape[0]
