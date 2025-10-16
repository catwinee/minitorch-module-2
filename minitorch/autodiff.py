from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Set, Dict

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    vals_1 = vals[:arg] + (vals[arg] + epsilon,) + vals[arg + 1 :]
    vals_2 = vals[:arg] + (vals[arg] - epsilon,) + vals[arg + 1 :]
    return (f(*vals_1) - f(*vals_2)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    ret: List[Variable] = list()
    visited_id_set: Set[Int] = set()

    def visit(variable: Variable) -> None:
        if variable.unique_id in visited_id_set:
            return
        visited_id_set.add(variable.unique_id)
        if not variable.is_leaf():
            for parent in variable.parents:
                visit(parent)
        ret.append(variable)

    visit(variable)
    return ret


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.

    ## Naive implementation using recurrence
    # for var, der in variable.chain_rule(deriv):
    #     if var.is_leaf():
    #         var.accumulate_derivative(der)
    #     else:
    #         backpropagate(var, der)

    # interactive implementation:
    order = reversed(list(topological_sort(variable)))
    grad_map: Dict[Int, float] = {}
    grad_map[variable.unique_id] = deriv
    for node in order:
        if node.is_leaf():
            node.accumulate_derivative(grad_map[node.unique_id])
        else:
            for parent, der in node.chain_rule(grad_map[node.unique_id]):
                if parent.unique_id not in grad_map:
                    grad_map[parent.unique_id] = 0.0
                grad_map[parent.unique_id] += der


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
