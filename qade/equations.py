from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from numpy.typing import ArrayLike

from qade.bases import Basis
from qade.functions import Formula
from qade.regression import (
    AnnealSchedule,
    ClassicalMinimizer,
    Encoding,
    Fitter,
    FittingParameters,
    Quadric,
)


@dataclass
class Equation:
    samples: np.ndarray
    coeffs: List[np.ndarray]
    bias: np.ndarray

    def homogeneous_coefficient(self, basis: Basis) -> np.ndarray:
        """
        For an equation of the form

            E[y](x) = b(x) + sum_k c_k(x) d^k[y(x)],

        compute the coefficient H_aA for the homogeneous part, defined by

            E[w_A Phi_A](x_a) = w_A H_aA + b_a,

        for a given basis of functions Phi_A(x).
        """
        (n_data, n_in) = self.samples.shape
        n_out = self.coeffs[0].shape[1]
        basis_dim = basis.dimension(self.samples.shape[1])
        out_structured = np.zeros((n_data, n_out, basis_dim))

        for k, c_k in enumerate(self.coeffs):
            c_k = c_k.reshape(n_data, n_out, n_in ** k)
            d_k = basis.derivatives(k, self.samples).reshape(
                n_data, basis_dim, n_in ** k
            )

            out_structured += np.einsum("aij,apj->aip", c_k, d_k)

        return out_structured.reshape(n_data, n_out * basis_dim)

    def quadric(self, basis: Basis) -> Quadric:
        """The quadric in the w_A variables given by sum_a (w_A H_aA + b_a)^2."""
        H = self.homogeneous_coefficient(basis)
        n_samples = self.samples.shape[0]
        return Quadric(
            J=(np.einsum("aA,aB->AB", H, H) / n_samples),
            h=(2 * np.einsum("aA,a->A", H, self.bias) / n_samples),
            offset=(self.bias @ self.bias / n_samples),
        )

    def n_out(self) -> int:
        return self.coeffs[0].shape[1]

    def n_in(self) -> int:
        return self.samples.shape[1]


@dataclass
class Solution:
    basis: Basis
    weights: np.ndarray
    loss: float

    def __call__(self, samples: ArrayLike):
        """Evaluate the solution at the given sample points."""
        return self.basis(samples) @ self.weights.T


def equation(formula: Formula, samples: ArrayLike) -> Equation:
    """
    Creates an Equation, representing any linear differential equation or boundary
    condition. The arguments are:

    - formula. The expression for the equation, constructed as a linear combination of
      derivatives of Function objects, plus the optional inhomogeneous term.

    - samples, an array-like object of shape (n_samples, n_in) (or just (n_samples,),
      if n_in == 1). Defines the sample points from the domain in which the equation
      must hold.
    """
    samples = np.atleast_1d(samples)
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    n_samples = samples.shape[0]

    return Equation(
        samples=samples,
        coeffs=[np.broadcast_to(c, (n_samples,) + c.shape[1:]) for c in formula.coeffs],
        bias=np.broadcast_to(formula.bias, (n_samples,)),
    )


def loss(equations: List[Equation], basis: Basis):
    """
    Returns the (J, h) parameters of the quadratic loss function Q(w) = w^T J w + h^T w,
    where w is the flattened array of the continuous weights. The arguments are

    - equations: List[Equation]. The equations to be included.

    - basis: Basis. The basis to be used in their solution.
    """
    quadric = sum((equation.quadric(basis) for equation in equations), Quadric.zero())
    return quadric.J, quadric.h


def normalize_encoding_parameters(
    input_: Optional[ArrayLike], basis_dimension: int, n_out: int
) -> np.ndarray:
    """
    Construct an array shape (n_out * basis_dimension,) of centers or scales for the
    binary encoding from the user input"""
    out = np.array(input_)
    if out.ndim == 0:
        out = out * np.ones(n_out * basis_dimension)

    return out.flatten()


def ising(
    equations: List[Equation],
    basis: Basis,
    n_spins: int = 3,
    centers: ArrayLike = 0,
    scales: ArrayLike = 1,
):
    """
    Returns the (J, h) parameters of the Ising model Hamiltonian H(w) = w^T J w + h^T w,
    where w is the flattened array of spins. The arguments are:

    - equations: List[Equation]. The equations to be included.

    - basis: Basis. The basis to be used in their solution.

    - n_spins: int = 3. The number of spins per weight in the binary encoding.

    - centers: ArrayLike = 0. The central values in the binary encoding of the weights.
      Must be a scalar or an array of size n_out * basis_dim.

    - scales: ArrayLike = 1. The scales in the binary encoding of the weights. Must be a
      scalar or an array of size n_out * basis_dim.
    """

    centers = normalize_encoding_parameters(
        centers, basis.dimension(equations[0].n_in()), equations[0].n_out()
    )
    scales = normalize_encoding_parameters(
        scales, basis.dimension(equations[0].n_in()), equations[0].n_out()
    )

    encoding = Encoding(n_spins, centers, scales)
    quadric = sum((equation.quadric(basis) for equation in equations), Quadric.zero())
    ising = encoding.ising(quadric)
    return ising.J, ising.h


def solve(
    equations: List[Equation],
    basis: Basis,
    n_spins: int = 3,
    n_epochs: int = 10,
    scale_factor: float = 0.5,
    anneal_schedule: Optional[AnnealSchedule] = None,
    num_reads: int = 200,
    qpu_solver: str = "Advantage_system4.1",
    centers: ArrayLike = 0,
    scales: ArrayLike = 1,
    verbose: bool = True,
    classical_minimizer: Optional[ClassicalMinimizer] = None,
):
    """
    Solves the given equations and returns a [Solution](#solution) object. The
    arguments are:

    - equations: List[Equation]. The equations to be solved, including initial/boundary
      conditions.

    - basis: Basis. The basis to be used in their solution.

    - n_spins: int = 3. The number of spins per weight in the binary encoding.

    - n_epochs: int = 10. The number of epochs in which the problem is solved, using the
      values obtained in the previous epoch as the center values in the binary encoding
      of the weights.

    - scale_factor: float = 0.5. The relative change in the scale in the binary encoding
      of the weights from one epoch to the next.

    - anneal_schedule: Optional[List[Tuple[int, int]]] = None. The schedule to be used
      by the quantum annealer. See D-Wave Ocean Tools' documentation for details. If
      None, it will default to a linear schedule of the form [(0, 0), (200, 0)].

    - num_reads: int = 200. The number of reads to be performed by the quantum annealer.

    - qpu_solver: str = "Advantage_system4.1". The QPU to be used.

    - centers: ArrayLike = 0. The initial central values in the binary encoding of the
      weights. Must be a scalar or an array of size n_out * basis_dim.

    - scales: ArrayLike = 1. The initial scales in the binary encoding of the weights.
      Must be a scalar or an array of size n_out * basis_dim.

    - verbose: bool = False. Determines if information about intermediate steps should
      be printed.

    - classical_minimizer:
      Optional[Callable[[Callable[[np.ndarray], float]], np.ndarray]] = None. If not
      None, a classical function to be used in place of the quantum procedure for
      minimizing the loss function, for testing purposes.
    """

    if anneal_schedule is None:
        anneal_schedule = [(0, 0), (200, 1)]

    centers = normalize_encoding_parameters(
        centers, basis.dimension(equations[0].n_in()), equations[0].n_out()
    )
    scales = normalize_encoding_parameters(
        scales, basis.dimension(equations[0].n_in()), equations[0].n_out()
    )

    parameters = FittingParameters(
        n_spins=n_spins,
        n_epochs=n_epochs,
        scale_factor=scale_factor,
        anneal_schedule=anneal_schedule,
        num_reads=num_reads,
        qpu_solver=qpu_solver,
        centers=centers,
        scales=scales,
        verbose=verbose,
        classical_minimizer=classical_minimizer,
    )
    quadric = sum((equation.quadric(basis) for equation in equations), Quadric.zero())
    components = Fitter(quadric).fit(parameters)

    return Solution(
        basis,
        components.reshape(equations[0].n_out(), -1),
        quadric(components),
    )
