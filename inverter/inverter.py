import numpy as np
from typing import Optional


# noinspection PyPep8Naming, PyShadowingNames
def inv_from_inv(A_inv: np.ndarray,
                 x: np.ndarray,
                 pivot: int) -> Optional[np.ndarray]:
    #
    L: np.ndarray = A_inv.dot(x)  # (a11, a21, a31)

    if _is_invertible(L, pivot):
        coef: int = (-1.0) / L.flat[pivot]
        L[pivot] = -1  # (a11, -1, a31)
        L *= coef  # (-a11/a21, 1/a21, -a31/a21)

        return _calculate_new_inv(A_inv, L, pivot)

    else:
        return None


# noinspection PyPep8Naming, PyShadowingNames
def _calculate_new_inv(A_inv: np.ndarray,
                       L: np.ndarray,
                       pivot: int) -> np.ndarray:
    #
    n = len(A_inv[0])
    target_matr = A_inv.copy()
    Q = np.eye(n)
    Q[:, pivot] = L.ravel()

    for i in range(n):
        for j in range(n):
            if i != pivot:
                target_matr[i, j] += Q[i, pivot] * A_inv[pivot, j]
            else:
                target_matr[i, j] = Q[i, pivot] * A_inv[pivot, j]

    return target_matr


# noinspection PyPep8Naming, PyShadowingNames
def _is_invertible(L: np.ndarray,
                   pivot: int) -> bool:
    #
    if not L[pivot]:
        print("[Inverter error] Matrix is not invertible.")

    return L[pivot] != 0
