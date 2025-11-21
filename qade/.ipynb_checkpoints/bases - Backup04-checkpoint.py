from abc import ABC
from dataclasses import dataclass
from typing import Callable
import numpy as np
from numpy.polynomial.chebyshev import Chebyshev

def binomial(m: int, n: int) -> int:
    return np.math.factorial(m) / (np.math.factorial(m - n) * np.math.factorial(n))

def chebyshev_polynomials(size_per_dim: int):
    return [Chebyshev.basis(deg) for deg in range(size_per_dim)]

def chebyshev_derivatives(size_per_dim: int, order: int):
    polys = chebyshev_polynomials(size_per_dim)
    return [poly.deriv(order) for poly in polys]

class Basis(ABC):
    def derivatives(self, k: int, x: np.ndarray) -> np.ndarray:
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.derivatives(0, x)

    def dimension(self, n_in: int) -> int:
        pass

@dataclass
class TensorProductBasis(Basis):
    name: str
    size_per_dim: int
    dphi: Callable[[int, int, np.ndarray], np.ndarray]

    def dimension(self, n_in: int) -> int:
        return self.size_per_dim ** n_in

    def __str__(self):
        return f"basis({self.name}, {self.size_per_dim})"

    __repr__ = __str__

    @staticmethod
    def fourier(size_per_dim: int) -> "TensorProductBasis":
        def dphi(k: int, j: int, x: np.ndarray):
            factor = 2 * np.pi * np.ceil(j / 2)
            return factor ** k * np.cos(factor * x + np.pi / 2 * (k - j % 2))

        return TensorProductBasis("fourier", size_per_dim, dphi)

    @staticmethod
    def chebyshev(size_per_dim: int) -> "TensorProductBasis":
        def dphi(k: int, j: int, x: np.ndarray):
            polys = chebyshev_derivatives(size_per_dim, k)
            return polys[j](x)

        return TensorProductBasis("chebyshev", size_per_dim, dphi)

    @staticmethod
    def monomial(size_per_dim: int) -> "TensorProductBasis":
        def dphi(k: int, j: int, x: np.ndarray) -> np.ndarray:
            if k > j:
                return np.array(0)
            return np.math.factorial(j) / np.math.factorial(j - k) * x ** (j - k)

        return TensorProductBasis("monomial", size_per_dim, dphi)

    @staticmethod
    def trig(size_per_dim: int) -> "TensorProductBasis":
        def psi(j: int, t: np.ndarray) -> np.ndarray:
            if j < 0 or j >= size_per_dim:
                return np.array(0)
            return np.cos(t) ** (size_per_dim - j - 1) * np.sin(t) ** j

        def dpsi(k: int, j: int, t: np.ndarray) -> np.ndarray:
            if j < 0 or j >= size_per_dim or k == 0:
                return psi(j, t)
            return -(size_per_dim - j - 1) * dpsi(k - 1, j + 1, t) + j * dpsi(k - 1, j - 1, t)

        def dphi(k: int, j: int, x: np.ndarray) -> np.ndarray:
            return (
                (np.pi / 2) ** k
                * binomial(size_per_dim - 1, j)
                * dpsi(k, j, np.pi / 2 * x)
            )

        return TensorProductBasis("trig", size_per_dim, dphi)

    def derivatives(self, k: int, x: np.ndarray) -> np.ndarray:
        x = np.atleast_1d(x)
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        (n_data, n_in) = x.shape
        feature_shape = (self.size_per_dim,) * n_in
        derivative_shape = (n_in,) * k
        out_structured = np.ones((n_data,) + feature_shape + derivative_shape)

        for derivative_indices in np.ndindex(derivative_shape):
            for feature_indices in np.ndindex(feature_shape):
                s = (slice(n_data), *feature_indices, *derivative_indices)

                for n, j in enumerate(feature_indices):
                    k = derivative_indices.count(n)
                    out_structured[s] *= self.dphi(k, j, x[:, n])  # type: ignore

        return out_structured.reshape(
            (n_data, self.size_per_dim ** n_in) + derivative_shape
        )


@dataclass
class RadialBasis(Basis):
    name: str
    size_per_dim: int
    scale: float
    dphi: Callable[[int, np.ndarray], np.ndarray]
    points: np.ndarray

    def dimension(self, n_in: int) -> int:
        return self.points.shape[0]

    def __str__(self):
        return f"basis({self.name}, {self.size_per_dim}, {self.scale})"

    __repr__ = __str__

    @staticmethod
    def gaussian(size_per_dim: int, scale: float, n_in: int) -> "RadialBasis":
        def dphi(k: int, r: np.ndarray) -> np.ndarray:
            phi = np.exp(-((r / scale) ** 2))
            order_tuple = (0,) * k + (1,)
            return (-scale) ** (-k) * hermval(r / scale, order_tuple) * phi

        axes = [np.linspace(0, 1, size_per_dim)] * n_in
        points = np.vstack(
            [coordinate.flatten() for coordinate in np.meshgrid(*axes)]
        ).T

        return RadialBasis("gaussian", size_per_dim, scale, dphi, points)

    @staticmethod
    def multiquadric(size_per_dim: int, scale: float, n_in: int) -> "RadialBasis":
        def dphi(k, r):
            if k == 0:
                return np.sqrt(r ** 2 + scale ** 2)
            elif k == 1:
                return r / np.sqrt(r ** 2 + scale ** 2)
            elif k == 2:
                return -(r ** 2) / np.sqrt(r ** 2 + scale ** 2) ** 3 + 1 / np.sqrt(
                    r ** 2 + scale ** 2
                )
            else:
                raise NotImplementedError

        axes = [np.linspace(0, 1, size_per_dim)] * n_in
        points = np.vstack(
            [coordinate.flatten() for coordinate in np.meshgrid(*axes)]
        ).T

        return RadialBasis("multiquadric", size_per_dim, scale, dphi, points)

    def derivatives(self, k: int, x: np.ndarray) -> np.ndarray:
        x = np.atleast_1d(x)
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        (n_data, n_in) = x.shape
        n_out = self.dimension(x.shape[1])
        out = np.empty((n_data, n_out) + (n_in,) * k)

        for out_index, point in enumerate(self.points):
            delta = x - point
            radius = np.sqrt(np.sum(delta ** 2, axis=1))

            for der_indices in np.ndindex((n_in,) * k):
                out_slice = (slice(n_data), out_index) + der_indices
                out[out_slice] = self.dphi(k, radius)

                for der_index in der_indices:
                    out[out_slice] *= np.divide(
                        delta[:, der_index],
                        radius,
                        where=(radius != 0),
                        out=np.ones_like(radius),
                    )

        return out


def basis(name: str, size_per_dim: int, scale: float = 1.0, n_in: int = 1):
    if name == "monomial":
        return TensorProductBasis.monomial(size_per_dim)
    elif name == "fourier":
        return TensorProductBasis.fourier(size_per_dim)
    elif name == "trig":
        return TensorProductBasis.trig(size_per_dim)
    elif name == "gaussian":
        return RadialBasis.gaussian(size_per_dim, scale, n_in)
    elif name == "multiquadric":
        return RadialBasis.multiquadric(size_per_dim, scale, n_in)
    elif name == "chebyshev":
        return TensorProductBasis.chebyshev(size_per_dim)
    else:
        raise Exception(f"Unknown basis '{name}'")