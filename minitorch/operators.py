"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(a: float, b: float) -> float:
    """Multiplies two numbers."""
    return a * b


def id(a: float) -> float:
    """Returns the input unchanged."""
    return a


def add(a: float, b: float) -> float:
    """Adds two numbers."""
    return a + b


def neg(a: float) -> float:
    """Negates a number."""
    return -a


def lt(a: float, b: float) -> bool:
    """Checks if one number is less than another."""
    return a < b


def eq(a: float, b: float) -> bool:
    """Checks if two numbers are equal."""
    return a == b


def max(a: float, b: float) -> float:
    """Returns the larger of two numbers."""
    return a if a > b else b


def is_close(a: float, b: float) -> bool:
    """Checks if two numbers are close in value.

    Returns:
        True if the absolute value of the two numbers is less than 0.01.

    """
    return abs(a - b) < 1e-2


# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
def sigmoid(a: float) -> float:
    """Calculates the sigmoid function.

    Returns:
        The output of the sigmoid function, a float value between 0 and 1.

    """
    if a >= 0:
        return 1.0 / (1.0 + math.exp(-a))
    else:
        return math.exp(a) / (1.0 + math.exp(a))


def relu(a: float) -> float:
    """Applies the ReLU activation function.

    Returns:
        The input if it is positive, otherwise 0.

    """
    return max(0.0, a)


def log(a: float) -> float:
    """Calculates the natural logarithm.

    Args:
        a: A non-zero number.

    Raises:
        ValueError: if the input is not positive.

    """
    if a <= 0:
        raise ValueError("Can not compute the logarithm of a non-positive number")
    return math.log(a)


def exp(a: float) -> float:
    """Calculates the exponential function."""
    return math.exp(a)


def inv(a: float) -> float:
    """Calculates the reciprocal.

    Args:
        a (float): A non-zero number.

    Raises:
        ValueError: if the input is zero.

    """
    if a == 0:
        raise ValueError("Can not compute the reciprocal of zero")
    return 1.0 / a


def log_back(a: float, d: float) -> float:
    """Computes the derivative of log times a second arg.

    Args:
        a (float): The input to the original log function.
        d (float): The upstream gradient.

    Returns:
        The downstream gradient d / a.

    """
    return d / a


def inv_back(a: float, d: float) -> float:
    """Computes the derivative of reciprocal times a second arg.

    Args:
        a (float): The input to the original inv function.
        d (float): The upstream gradient.

    Returns:
        The downstream gradient -d * a ** -2.

    """
    return -d * a**-2


def relu_back(a: float, d: float) -> float:
    """Computes the derivative of ReLU times a second arg.

    Args:
        a (float): The input to the original relu function.
        d (float): The upstream gradient.

    Returns:
        The downstream gradient. 0 if the input is negative, d otherwise.

    """
    return 0 if a < 0 else d


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher-order function that applies a given function to each element of an iterable."""

    def apply(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for it in ls:
            ret.append(fn(it))
        return ret

    return apply


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher-order function that combines elements from two iterables using a given function."""

    def apply(ls_1: Iterable[float], ls_2: Iterable[float]) -> Iterable[float]:
        ret = []
        for it_1, it_2 in zip(ls_1, ls_2):
            ret.append(fn(it_1, it_2))
        return ret

    return apply


def reduce(fn: Callable[[float, float], float]) -> Callable[[Iterable[float]], float]:
    """Higher-order function that reduces an iterable to a single value using a given function.

    Raises:
        ValueError: If the input Iterable is empty.

    """

    def apply(ls: Iterable[float]) -> float:
        iterator = iter(ls)

        try:
            val = next(iterator)
        except StopIteration:
            return 0
            # raise ValueError("Cannot reduce an empty Iterable.")
            # Unfortunately, pytest doesn't want me to raise errors

        for it in iterator:
            val = fn(it, val)
        return val

    return apply


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list using map."""
    return map(neg)(ls)


def addLists(ls_1: Iterable[float], ls_2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two lists using zipWith."""
    return zipWith(add)(ls_1, ls_2)


def sum(ls: Iterable[float]) -> float:
    """Sum all elements in a list using reduce."""
    return reduce(add)(ls)


def prod(ls: Iterable[float]) -> float:
    """Calculate the product of all elements in a list using reduce."""
    return reduce(mul)(ls)
