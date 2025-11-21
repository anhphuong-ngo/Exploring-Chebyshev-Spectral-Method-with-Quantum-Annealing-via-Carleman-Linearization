from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import dimod
import neal
from dwave.system import DWaveSampler, LazyFixedEmbeddingComposite
import numpy as np
from scipy.optimize import minimize

@dataclass
class Quadric:
    h: np.ndarray
    J: np.ndarray
    offset: float = 0

    @staticmethod
    def regression_loss(domain: np.ndarray, targets: np.ndarray) -> "Quadric":
        """Sum-of-squares loss for linear regression (without constant terms)."""
        return Quadric(h=(-2 * targets @ domain), J=(domain.T @ domain))

    @staticmethod
    def zero():
        return Quadric(h=np.array(0), J=np.array(0))

    def __call__(self, vector: np.ndarray) -> float:
        """Evaluate the quadric at a given vector."""
        vector = vector.flatten()
        return vector @ self.J @ vector + self.h @ vector + self.offset

    def __add__(self, other: "Quadric") -> "Quadric":
        return Quadric(
            h=(self.h + other.h),
            J=(self.J + other.J),
            offset=(self.offset + other.offset),
        )

@dataclass
class Encoding:
    n_spins: int
    centers: np.ndarray
    scales: np.ndarray

    def epsilons(self) -> np.ndarray:
        """Vector containing inverse powers of 2: [1, 1/2, 1/4, ...]."""
        return np.array([2 ** (-alpha) for alpha in range(self.n_spins)])

    def ising(self, quadric: Quadric) -> Quadric:
        (dim,) = quadric.h.shape
        eps = self.epsilons()

        # Intermediate arrays with separate bits and quadric indices
        J_hat = np.einsum(
            "AB,A,B,a,b->aAbB", quadric.J, self.scales, self.scales, eps, eps
        )
        h_hat = np.einsum("A,A,a->aA", quadric.h, self.scales, eps) + 2 * np.einsum(
            "AB,B,A,a->aA", quadric.J, self.centers, self.scales, eps
        )

        # Merge spin and quadric indices
        return Quadric(
            h_hat.reshape(self.n_spins * dim),
            J_hat.reshape(self.n_spins * dim, self.n_spins * dim),
        )

    def encode(self, quadric: Quadric) -> dimod.BQM:
        """Convert a continuous quadric to a binary one."""
        ising = self.ising(quadric)

        # Merge spin and quadric indices
        return dimod.BQM(ising.h, ising.J, quadric(self.centers), dimod.SPIN)

    def decode(self, spins: np.ndarray) -> np.ndarray:
        """Convert a flat array of bits to the corresponding continuous array."""
        return self.centers + self.scales * (
            self.epsilons() @ spins.reshape(self.n_spins, -1)
        )

AnnealSchedule = List[Tuple[int, float]]
LossFunction = Callable[[np.ndarray], float]
ClassicalMinimizer = Callable[[LossFunction], np.ndarray]

@dataclass
class FittingParameters:
    n_spins: int
    n_epochs: int
    scale_factor: float
    anneal_schedule: AnnealSchedule
    num_reads: int
    qpu_solver: str
    centers: np.ndarray
    scales: np.ndarray
    verbose: bool
    classical_minimizer: Optional[ClassicalMinimizer]

@dataclass
class Fitter:
    def __init__(self, quadric):
        self.quadric = quadric

    def fit(self, parameters):
        """
        Iterative minimization of self.quadric by setting the center values of the
        encoding to the results of the previous iteration, and rescaling the
        corresponding scales.
        """
        encoding = Encoding(parameters.n_spins, parameters.centers, parameters.scales)
        sampler = neal.SimulatedAnnealingSampler()

        for epoch in range(parameters.n_epochs):
            sampleset = sampler.sample(encoding.encode(self.quadric), num_reads=parameters.num_reads)
            best_sample = sampleset.first.sample
            centers = np.array([best_sample[i] for i in range(len(encoding.centers))])

            encoding.centers = centers
            encoding.scales /= parameters.scale_factor

            if parameters.verbose:
                print(f"epoch = {epoch}, loss = {self.quadric(encoding.centers)}")

        if parameters.classical_minimizer:
            return self.classical_minimization(parameters, encoding.centers)

        return encoding.centers

    def classical_minimization(self, parameters, initial_centers):
        # Define the objective function to be minimized
        def objective(weights):
            return self.quadric(weights)

        # Perform the optimization
        result = minimize(objective, initial_centers.flatten(), method='SLSQP')
        return result.x.reshape(initial_centers.shape)