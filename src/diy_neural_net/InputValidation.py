from typing import List, Union, Tuple, Any, TYPE_CHECKING

from .DeviceSelector import ArrayType

if TYPE_CHECKING:
    from .Losses import Loss


class InputValidator:
    """Static utility class for validating neural network inputs and parameters.

    Provides comprehensive validation for various neural network components
    including layers, parameters, and data shapes.
    """

    @staticmethod
    def validate_n_classes(n_classes: int) -> int:
        """Validate number of classes parameter.

        Args:
            n_classes: Number of output classes

        Returns:
            Validated number of classes

        Raises:
            TypeError: If n_classes is not an integer
            ValueError: If n_classes is not strictly positive
        """
        if not isinstance(n_classes, int):
            raise TypeError("n_classes needs to be an int")

        if n_classes < 1:
            raise ValueError("n_classes needs to be strictly positive")

        return n_classes

    @staticmethod
    def validate_layers(layers: Union[List, Tuple]) -> List[Any]:
        """Validate list of neural network layers.

        Args:
            layers: List or tuple of layer objects

        Returns:
            Validated list of layers

        Raises:
            TypeError: If layers is not a non-empty list/tuple or contains invalid types
            ValueError: If layer dimensions don't match
        """
        from .Layers import Layer, Dropout
        from .Activations import Activation

        if not isinstance(layers, (list, tuple)) or len(layers) < 1:
            raise TypeError("Layers needs to be a non empty list or a tuple")

        if not all(isinstance(layer, (Layer, Activation)) for layer in layers):
            raise TypeError("Layers must consist of Layers or Activations only")

        layers_only = [
            layer
            for layer in layers
            if isinstance(layer, Layer) and not isinstance(layer, Dropout)
        ]

        for i in range(len(layers_only) - 1):
            if not layers_only[i].output_size == layers_only[i + 1].input_size:
                raise ValueError("Matrix shapes do not match")

        return layers

    @staticmethod
    def validate_learning_rate(learning_rate: Union[float, int]) -> Union[float, int]:
        """Validate learning rate parameter.

        Args:
            learning_rate: Learning rate value

        Returns:
            Validated learning rate

        Raises:
            TypeError: If learning_rate is not float or int
            ValueError: If learning_rate is not strictly positive
        """
        if not isinstance(learning_rate, (float, int)):
            raise TypeError("Learning Rate needs to be a float or an int")

        if learning_rate <= 0:
            raise ValueError("Learning rate needs to be strictly positive")

        return learning_rate

    @staticmethod
    def validate_criterion(criterion) -> "Loss":
        """Validate loss function criterion.

        Args:
            criterion: Loss function object

        Returns:
            Validated loss function

        Raises:
            ValueError: If criterion is not a valid Loss instance
        """
        from .Losses import Loss

        if not isinstance(criterion, Loss):
            raise ValueError("Loss used must be a valid Loss")

        return criterion

    @staticmethod
    def validate_keep_prob(keep_prob: Union[int, float]) -> Union[int, float]:
        """Validate dropout keep probability.

        Args:
            keep_prob: Probability of keeping neurons (0 < keep_prob <= 1)

        Returns:
            Validated keep probability

        Raises:
            TypeError: If keep_prob is not int or float
            ValueError: If keep_prob is not in valid range
        """
        if not isinstance(keep_prob, (int, float)):
            raise TypeError("keep prob should be an int or a float")

        if keep_prob <= 0 or keep_prob > 1:
            raise ValueError(
                "keep_prob needs to be between 0 exclusive and 1 inclusive"
            )

        return keep_prob

    @staticmethod
    def validate_number_units(number_of_units: int) -> int:
        """Validate number of units in a layer.

        Args:
            number_of_units: Number of units/neurons in layer

        Returns:
            Validated number of units

        Raises:
            TypeError: If number_of_units is not an integer
            ValueError: If number_of_units is not greater than 0
        """
        if not isinstance(number_of_units, int):
            raise TypeError("Number of units needs to be an int")

        if number_of_units <= 0:
            raise ValueError("Number of units needs to greater than 0")

        return number_of_units

    @staticmethod
    def validate_delta(delta: Union[float, int]) -> Union[float, int]:
        """Validate delta parameter for early stopping.

        Args:
            delta: Minimum improvement threshold

        Returns:
            Validated delta value

        Raises:
            TypeError: If delta is not float or int
            ValueError: If delta is negative
        """
        if not isinstance(delta, (float, int)):
            raise TypeError("Delta needs to be a float or an int")

        if delta < 0:
            raise ValueError("Delta needs to positive")

        return delta

    @staticmethod
    def validate_patience(patience: int) -> int:
        """Validate patience parameter for early stopping.

        Args:
            patience: Number of epochs to wait before stopping

        Returns:
            Validated patience value

        Raises:
            TypeError: If patience is not an integer
            ValueError: If patience is negative
        """
        if not isinstance(patience, int):
            raise TypeError("Patience needs to be an int")

        if patience < 0:
            raise ValueError("Patience needs to positive")

        return patience

    @staticmethod
    def validate_same_shape(a: ArrayType, b: ArrayType) -> Tuple[ArrayType, ArrayType]:
        """Validate that two arrays have the same shape and type.

        Args:
            a: First array
            b: Second array

        Returns:
            Tuple of validated arrays

        Raises:
            TypeError: If arrays are different types
            ValueError: If array shapes don't match
        """
        if type(a) is not type(b):
            raise TypeError(
                "paramater 1 and 2 are two different types, they need to be of same type"
            )

        if a.shape != b.shape:
            raise ValueError("paramater 1 and 2's shapes do not match")

        return a, b
