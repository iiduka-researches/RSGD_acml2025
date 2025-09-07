import numpy as np

from .manifold import Manifold


class Sphere(Manifold):
    def __init__(self, n: int) -> None:
        super(Sphere, self).__init__()
        self.n = n

    def __repr__(self) -> str:
        return f'S^{self.n - 1}'

    def dim(self) -> int:
        return self.n - 1

    def projection(
        self, 
        point: np.ndarray,
        vector: np.ndarray
    ) -> np.ndarray:
        return vector - (point @ vector) * point

    def retraction(
        self, 
        point: np.ndarray,
        tangent_vector: np.ndarray
    ) -> np.ndarray:
        return (point + tangent_vector) / np.linalg.norm(point + tangent_vector)