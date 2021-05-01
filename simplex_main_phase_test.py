import numpy as np
from simplex_main_phase import simplex_main_phase


if __name__ == '__main__':
    # c = np.array([10, 11, 0, 0, 1, 0], dtype = float)
    # A = np.array(
    #   [
    #     [3, 2, 1, 0, 0, 0],
    #     [2, 5, 0, 1, 0, 0],
    #     [0, 1, 0, 0, 1, 0],
    #     [1, 0, 0, 0, 0, 1]
    #   ],
    #   dtype = float
    # )
    # x = np.array([0, 0, 21, 25, 4, 6], dtype = float)
    # J_b = np.array([2, 3, 4, 5])

    # c = np.array([19, 29, -1, 0, 0, 0], dtype = float)
    # A = np.array(
    #   [
    #     [1, 1, 1, 0, 0, 0],
    #     [2, 5, 0, 1, 0, 0],
    #     [0, 1, 0, 0, 1, 0],
    #     [1, 0, 0, 0, 0, 1]
    #   ],
    #   dtype = float
    # )
    # x = np.array([0, 0, 9, 30, 5, 7], dtype = float)
    # J_b = np.array([2, 3, 4, 5])

    # c = np.array([20, 26, 0, 0, 0, 1], dtype = float)
    # A = np.array(
    #   [
    #     [1, 2, 1, 0, 0, 0],
    #     [2, 1, 0, 1, 0, 0],
    #     [1, 0, 0, 0, 1, 0],
    #     [0, 1, 0, 0, 0, 1]
    #   ],
    #   dtype = float
    # )
    # x = np.array([2, 4, 0, 3, 3, 0], dtype = float)
    # J_b = np.array([4, 1, 0, 3])

    # c = np.array([1, 1], dtype = float)
    # A = np.array(
    #   [
    #     [1, 2],
    #     [2, 1]
    #   ],
    #   dtype = float
    # )
    # x = np.array([1, 1], dtype = float)
    # J_b = np.array([1, 0])

    # c = np.array([1, 2], dtype = float)
    # A = np.array(
    #   [
    #     [1, -1]
    #   ],
    #   dtype = float
    # )
    # x = np.array([0, 0], dtype = float)
    # J_b = np.array([1])

    # c = np.array([20, 26, 0, 0, 0, 1], dtype = float)
    # A = np.array(
    #   [
    #     [1, 2, 1, 0, 0, 0],
    #     [2, 1, 0, 1, 0, 0],
    #     [1, 0, 0, 0, 1, 0],
    #     [0, 1, 0, 0, 0, 1]
    #   ],
    #   dtype = float
    # )
    # x = np.array([0, 0, 10, 11, 5, 4], dtype = float)
    # J_b = np.array([2, 3, 4, 5])

    # c = np.array([11, 21, 1, 0, 0, 0], dtype = float)
    # A = np.array(
    #   [
    #     [1, 1, 1, 0, 0, 0],
    #     [2, 5, 0, 1, 0, 0],
    #     [1, 0, 0, 0, 1, 0],
    #     [0, 1, 0, 0, 0, 1]
    #   ],
    #   dtype = float
    # )
    # x = np.array([0, 0, 10, 35, 9, 6], dtype = float)
    # J_b = np.array([2, 3, 4, 5])

    # c = np.array([25, 34, 1, 0, 0, 0], dtype = float)
    # A = np.array(
    #   [
    #     [5, 4, 1, 0, 0, 0],
    #     [3, 7, 0, 1, 0, 0],
    #     [1, 0, 0, 0, 1, 0],
    #     [0, 1, 0, 0, 0, 1]
    #   ],
    #   dtype = float
    # )
    # x = np.array([0, 0, 55, 56, 10, 7], dtype = float)
    # J_b = np.array([2, 3, 4, 5])

    c = np.array([10, 19, 0, 0, -1, 0], dtype = float)
    A = np.array(
      [
        [5, 3, 1, 0, 0, 0],
        [2, 5, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 1]
      ],
      dtype = float
    )
    x = np.array([0, 0, 40, 35, 6, 7], dtype = float)
    J_b = np.array([2, 3, 4, 5])

    simplex_main_phase(c, A, x, J_b)
