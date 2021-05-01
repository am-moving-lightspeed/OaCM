import numpy as np

from typing import Optional
from typing import Tuple

from simplex_main_phase import simplex_main_phase
from simplex_main_phase import invert_conditions_matrix
from simplex_main_phase import fetch_basis_matrix
from simplex_main_phase import pick_non_basis_indexes


def simplex_first_phase(A: np.ndarray,
                        b: np.ndarray) -> Tuple[Optional[np.ndarray],
                                                Optional[np.ndarray],
                                                Optional[np.ndarray]]:
    """
    * @:return x*   -- new basis plan;
    *          J_b* -- new basis indexes;
    *          A*   -- new/same conditions matrix.
    """

    """
    * A -- исходная матрица условий;
    * b -- вектор ограничений.
    """

    # Method start log.
    print("\n>>> Simplex method first phase has started.")

    m: int  # кол-во строк в матрице A.
    n: int  # кол-во столбцов в матрице A.
    c1: np.ndarray  # вектор вспом. целевой функции.
    A1: np.ndarray  # матрица условий вспом. целевой функции.
    x1: np.ndarray  # план вспом. целевой функции.
    J1: np.ndarray  # мн-во "искусственных" индексов.
    J_b1: np.ndarray  # базис.

    m = A.shape[0]
    n = A.shape[1]

    # Привести b к виду >= 0.
    for i in range(m):
        if b[i] < 0:
            b[i] *= -1
            A[i, :] *= -1

    # Привести задачу к виду
    # -e'x_u -> max, Ax + Ex_u = b, x >= 0, x_u >= 0.

    # c1 = (c, -e).
    c1 = np.array(
      [0 for _ in range(n)] +
      [-1 for _ in range(m)],
      dtype = float
    )

    # A1 = (A, E).
    E = np.eye(m)
    A1 = np.concatenate((A, E), axis = 1)

    # x1 = (x = 0, x_u = b)
    x1 = np.array(
      [0 for _ in range(n)] +
      [b[i] for i in range(m)],
      dtype = float
    )

    # J1 = {n, n + 1, ..., n + m}.
    J1 = np.array(
      [i for i in range(n, n + m)]
    )

    # Поиск решения через симплекс-метод.
    x1, J_b1 = simplex_main_phase(c1, A1, x1, J1.copy())

    result: int = _check_solution(x1, J_b1, J1, m, n)
    if result == -1 or result == 1:
        # Method end log.
        print(">>> Simplex method first phase has ended.\n")

        return None, None, None

    elif result == 2:
        # Method end log.
        print(">>> Simplex method first phase has ended.\n")

        return x1[:n], J_b1, A

    else:
        while True:
            k: int = _find_overlapping_element_index(J_b1, J1)

            if k != -1:
                J_n1 = pick_non_basis_indexes(J_b1, n)
                A_b1 = fetch_basis_matrix(A1, J_b1)  # баз-я матрица на основе нового баз-го плана.
                B1 = invert_conditions_matrix(A_b1)  # матрица, обратная базисной.
                e_k = np.array(
                  [0 if i != k else 1 for i in range(m)]  # ед-й в-р с 1 на k-ом месте.
                )

                alpha = e_k.dot(B1).dot(A1)
                for i in range(len(alpha)):  # in order to reduce floating-point-caused errors.
                    alpha[i] = round(alpha[i], 10)

                # Если не существует элемента alpha[i] != 0, то...
                if not _try_change_basis(alpha, k, J_b1, J_n1):
                    i0: int = J1[k] - n  # индекс линейно зависимого ограничения.

                    A = np.delete(A, i0, axis = 0)
                    A1 = np.delete(A1, i0, axis = 0)
                    A1 = np.delete(A1, J_b1[k], axis = 1)
                    x1 = np.delete(x1, J_b1[k], axis = 0)

                    J_b1 = np.delete(J_b1, k)

                    m -= 1

            else:
                # Method end log.
                print(">>> Simplex method first phase has ended.\n")

                return x1[:n], J_b1, A


def _try_change_basis(alpha: np.ndarray,
                      k: int,
                      J_b1: np.ndarray,
                      J_n1: np.ndarray) -> bool:
    """
    * alpha -- альфа-вектор;
    * k     -- позиция индекса из мн-ва J1;
    * J_b1  -- базисные индексы;
    * J_n1  -- небазисные индексы;
    """

    for i in range(len(J_n1)):
        if alpha[J_n1[i]] != 0:
            J_b1[k] = J_n1[i]
            return True

    return False


def _check_solution(x1: np.ndarray,
                    J_b1: np.ndarray,
                    J1: np.ndarray,
                    m: int,
                    n: int) -> int:
    """
    * x1   -- план вспом. целевой функции;
    * J_b1 -- базисные индексы;
    * J1   -- мн-во "искусственных" индексов;
    * m    -- кол-во строк в матрице A;
    * n    -- кол-во столбцов в матрице A.
    """

    """
    * -1 -- return code for simplex_main_phase() error;
    *  1 -- return code for no solutions possible (x_u != 0);
    *  2 -- return code for x_u = 0, J1 and J_b1 don't have common elements (solution found);
    *  3 -- return code otherwise;
    """

    # Check for simplex_main_phase() inner errors
    if x1 is None or J_b1 is None:
        return -1

    for i in range(n, n + m):
        if x1[i] != 0:
            print("Исходная задача решений не имеет.")

            return 1

    return 3 if _are_sets_overlapping(J_b1, J1) else 2


def _are_sets_overlapping(set1: np.ndarray,
                          set2: np.ndarray) -> bool:
    #
    for i in set1:
        if i in set2:
            return True

    return False


def _find_overlapping_element_index(set1: np.ndarray,
                                    set2: np.ndarray) -> int:
    #
    lower_set, greater_set = (set1, set2) if len(set1) <= len(set2) else (set2, set1)
    size = len(lower_set)

    for i in range(size):
        if lower_set[i] in greater_set:
            return i

    return -1
