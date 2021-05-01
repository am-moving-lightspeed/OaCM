import numpy as np
import numpy.linalg as lalg
import math

from typing import Optional
from typing import Tuple

from inverter import inv_from_inv


# noinspection PyPep8Naming
def simplex_main_phase(c: np.ndarray,
                       A: np.ndarray,
                       x: np.ndarray,
                       J_b: np.ndarray) -> Tuple[Optional[np.ndarray],
                                                 Optional[np.ndarray]]:
    """
    * c   -- вектор целевой функции;
    * A   -- матрица условий;
    * x   -- план;
    * J_b -- мн-во базисных индексов.
    """

    # Method start log.
    print("\n>>> Simplex method main phase has started")

    A_b: np.ndarray  # базисная матрица;
    B: np.ndarray  # матрица, обратная базисной матрице условий;
    c_b: np.ndarray  # компоненты целевой ф-ии по базисным индексам;
    u: np.ndarray  # вектор потенциалов;
    delta: np.ndarray  # вектор оценок;
    n: int  # кол-во столбцов;
    J_n: np.ndarray  # мн-во небазисных индексов;

    n = A.shape[1]
    J_n = pick_non_basis_indexes(J_b, n)

    while True:
        # Iteration log.
        print(f"x: {x}, basis indexes: {J_b}")

        # Нахождение базисной матрицы.
        A_b = fetch_basis_matrix(A, J_b)

        # Матрица, обратная базисной.
        B = invert_conditions_matrix(A_b)
        if B is None:
            # Method end log.
            print(">>> Simplex method main phase has ended.\n")

            return None, None

        # Баз-е комп-ты вектора целевой ф-ии.
        c_b = _pick_target_func_basis(c, J_b)

        # Вектор потенциалов.
        u = c_b.dot(B)
        # Вектор оценок.
        delta = u.dot(A) - c
        for i in range(len(delta)):
            delta[i] = round(delta[i], 10)  # in order to reduce floating-point-caused errors.

        # Проверка оценок на >= 0
        # (j0 такой, что delta_j0 < 0, либо None, если все delta >= 0).
        j0 = _find_delta_lower_than_zero(delta, J_n)

        if j0 is None:
            # Method end log.
            print(">>> Simplex method main phase has ended.\n")

            # Оптимальный план.
            for i in range(len(x)):
                x[i] = round(x[i], 10)  # in order to reduce floating-point-caused errors.
            return x, J_b

        # Вычисление и проверка вектора z.
        z = B.dot(A[:, j0])
        if _is_vector_equal_or_lower_than_zero(z):
            print("Решений нет: целевая функция не ограничена сверху.")

            # Method end log.
            print(">>> Simplex method main phase has ended.\n")

            return None, None

        # Вычисление тета и индекса s для замены.
        theta, s = _find_theta(z, x, J_b)

        # Обновление базисного и небазисного мн-в индексов.
        J_b[s] = j0
        J_n = pick_non_basis_indexes(J_b, n)

        _rebuild_basis_plan(x, z, J_n, J_b, theta, j0)


def fetch_basis_matrix(A: np.ndarray,
                       J_b: np.ndarray) -> np.ndarray:
    """
    * A   -- исходная матрица условий;
    * J_b -- мн-во базисных индексов.
    """

    m = len(J_b)
    A_b = np.empty((m, m), dtype = float)
    for i in range(m):
        A_b[:, i] = A[:, J_b[i]]

    return A_b


def invert_conditions_matrix(A_b: np.ndarray,
                             A_b_inv: Optional[np.ndarray] = None,
                             pivot: Optional[int] = None) -> Optional[np.ndarray]:
    """
    * A_b     -- исходная базисная матрица;
    * A_b_inv -- матрица, обратная исходной базисной;
    * pivot   -- индекс столбца для замены в единичной матрице.
    """

    if A_b_inv is not None:
        return inv_from_inv(A_b_inv, A_b[:, pivot], pivot)
    else:
        try:
            return lalg.inv(A_b)

        except lalg.LinAlgError as err:
            (
              print("[Simplex main phase error] Singular matrix.")
              if 'Singular matrix' in str(err) else
              print("[Simplex main phase error] Unknown.")
            )

            return None


def _find_delta_lower_than_zero(delta: np.ndarray,
                                J_n: np.ndarray) -> Optional[int]:
    """
    * delta -- вектор оценок;
    * J_n   -- мн-во небазисных индексов.
    """

    for i in J_n:
        if delta[i] < 0:
            return i

    return None


def _is_vector_equal_or_lower_than_zero(vector: np.ndarray) -> bool:
    #
    for value in vector:
        if value > 0:
            return False

    return True


def _find_theta(z: np.ndarray,
                x: np.ndarray,
                J_b: np.ndarray) -> Tuple[float, int]:
    """
    * x   -- текущий план;
    * J_b -- мн-во базисных индексов.
    """

    theta = math.inf
    tmp = math.inf
    position = 0

    for i in range(len(z)):
        if (
          z[i] > 0 and
          (tmp := x[J_b[i]] / z[i]) < theta
        ):
            theta = tmp
            position = i

    return theta, position


def _rebuild_basis_plan(x: np.ndarray,
                        z: np.ndarray,
                        J_n: np.ndarray,
                        J_b: np.ndarray,
                        theta: float,
                        j0: int) -> np.ndarray:
    """
    * x     -- текущий базисный план;
    * J_n   -- мн-во небазисных индексов;
    * J_b   -- мн-во базисных индексов;
    * theta -- тета;
    * j0    -- индекс, на который производится замена.
    """

    for i in range(len(z)):
        x[J_b[i]] = x[J_b[i]] - theta * z[i]
    for i in J_n:
        x[i] = 0
    x[j0] = theta

    return x


def _pick_target_func_basis(c: np.ndarray,
                            J_b: np.ndarray) -> np.ndarray:
    """
    * c   -- вектор целевой функции;
    * J_b -- мн-во базисных индексов.
    """

    m = J_b.size
    c_b = np.empty(m, dtype = float)
    for i in range(m):
        c_b[i] = c[J_b[i]]

    return c_b


def pick_non_basis_indexes(J_b: np.ndarray,
                           n: int) -> np.ndarray:
    """
    * J_b -- мн-во базисных индексов;
    * n   -- кол-во столбцов.
    """

    J = np.arange(n)
    J_n = []
    for item in J:
        if item not in J_b:
            J_n.append(item)

    return np.array(J_n)
