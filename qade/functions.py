from dataclasses import dataclass
from typing import List, Union, Sequence
from itertools import zip_longest

import numpy as np
from numpy.typing import ArrayLike


@dataclass
class Function:
    n_in: int
    n_out: int
    out_index: int

    def __str__(self):
        if self.n_out == 1:
            return f"function(n_in={self.n_in}, n_out={self.n_out})"
        return (
            f"function(n_in={self.n_in}, n_out={self.n_out}, "
            f"out_index={self.out_index})"
        )

    def __getitem__(self, orders: Union[int, Sequence[int]]) -> "Formula":
        """
        Construct a Formula representing partial derivatives. The expression
        f[n1, n2, ...] means that the n1 derivatives with respect to the first
        input variable, n2 derivatives with respect to the second input variable,
        etc. are taken.
        """
        if not isinstance(orders, Sequence):
            orders = [orders]

        order = sum(orders)
        indices = sum(((n,) * k for n, k in enumerate(orders)), ())

        coeffs = [
            np.zeros((1, self.n_out) + (self.n_in,) * k) for k in range(order + 1)
        ]
        coeffs[order][(0, self.out_index) + tuple(indices)] = 1

        return Formula(coeffs=coeffs, bias=np.array(0))


@dataclass
class Formula:
    """
    Represents linear combinations of derivatives of functions, plus an inhomogeneous
    term. The coeffs attribute gives the coefficients of the derivatives, in increasing
    order (from 0), and the bias attribute gives the inhomogeneous term.
    """

    coeffs: List[np.ndarray]
    bias: np.ndarray

    @staticmethod
    def constant(bias: ArrayLike) -> "Formula":
        return Formula(coeffs=[], bias=np.array(bias))

    __array_ufunc__ = None

    def __add__(self, other: Union[ArrayLike, "Formula"]) -> "Formula":
        if not isinstance(other, Formula):
            other = Formula.constant(other)
        return Formula(
            coeffs=[
                c1 + c2
                for c1, c2 in zip_longest(self.coeffs, other.coeffs, fillvalue=0)
            ],
            bias=(self.bias + other.bias),
        )

    __radd__ = __add__

    def __neg__(self) -> "Formula":
        return Formula(
            coeffs=[-c for c in self.coeffs],
            bias=-self.bias,
        )

    def __sub__(self, other: Union[ArrayLike, "Formula"]) -> "Formula":
        if not isinstance(other, Formula):
            other = np.array(other)
        return self + (-other)

    def __rsub__(self, other: Union[ArrayLike, "Formula"]) -> "Formula":
        if not isinstance(other, Formula):
            other = np.array(other)
        return (-self) + other

    def __mul__(self, other: ArrayLike) -> "Formula":
        other = np.array(other)
        return Formula(
            coeffs=[(other * c.T).T for c in self.coeffs],
            bias=(other * self.bias),
        )

    __rmul__ = __mul__

    def __truediv__(self, other: ArrayLike) -> "Formula":
        return (1 / other) * self


def function(n_in: int, n_out: int) -> Union[Function, List[Function]]:
    """
    Creates Function objects, to be used in definition of the equations. The arguments
    are:

    - n_in: int. The number of input variables of the function, i.e. the dimension of
      the domain.

    - n_out: int. The number of output variables of the function, i.e. the dimension of
      the target space. For `n_out == 1`, `qade.function` returns a single Function.
      For n_out > 1, it returns list of `Function` objects, one for each output
      component.

    A differential equation or boundary condition is then defined as a linear
    combination of the derivatives of the functions, plus possible an inhomogeneous
    term, with coefficients and inhomogeneous term being either scalars or flat
    array-like objects (numpy arrays, lists, ...) having the same length as the set of
    sample points at which the equation is evaluated. The examples show how this is done
    in different practical settings. The notation f[n1, n2, ...] represents the
    derivative of f with respect to the first variable n1 times, with respect to the
    second variable n2 times, etc.
    """

    if n_out == 1:
        return Function(n_in, n_out, 0)

    return [Function(n_in, n_out, out_index) for out_index in range(n_out)]
