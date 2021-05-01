import numpy as np
from simplex_first_phase import simplex_first_phase


if __name__ == '__main__':
    # 1.2.5
    # x, J_b, _ = simplex_first_phase(
    #   np.array(
    #     [
    #       [1, 1, -4, 0],
    #       [0, 1, 2, 1],
    #     ]
    #   ),
    #   np.array(
    #     [0, 6]
    #   )
    # )

    # 1.2.12
    # x, J_b, _ = simplex_first_phase(
    #   np.array(
    #     [
    #       [-1, 1, 1, 0],
    #       [4, 5, 0, 1],
    #       [10, 17, 2, 3]
    #     ]
    #   ),
    #   np.array(
    #     [2, 28, 88]
    #   )
    # )

    # 1.2.4
    # x, J_b, _ = simplex_first_phase(
    #   np.array(
    #     [
    #       [-3, 4, 1, 0],
    #       [5, 2, 0, 1]
    #     ]
    #   ),
    #   np.array(
    #     [8, 30]
    #   )
    # )

    # 1.2.1
    # x, J_b, _ = simplex_first_phase(
    #   np.array(
    #     [
    #       [-1, 3, 1, 0],
    #       [4, 3, 0, 1],
    #       [2, 9, 2, 1]
    #     ]
    #   ),
    #   np.array(
    #     [9, 24, 42]
    #   )
    # )

    # 1.2.18
    # x, J_b, _ = simplex_first_phase(
    #   np.array(
    #     [
    #       [-1, 3, 1, 0],
    #       [4, 1, 0, 1],
    #       [3, 4, 1, 1]
    #     ]
    #   ),
    #   np.array(
    #     [6, 28, 34]
    #   )
    # )

    # 1.2.19
    # x, J_b, _ = simplex_first_phase(
    #   np.array(
    #     [
    #       [-1, 5, 1, 0],
    #       [1, 1, 0, 1],
    #       [0, 6, 1, 1]
    #     ]
    #   ),
    #   np.array(
    #     [5, 4, 9]
    #   )
    # )

    # 1.2.30
    # x, J_b, _ = simplex_first_phase(
    #   np.array(
    #     [
    #       [-1, 3, 1, 0, 0],
    #       [4, 1, 0, 1, 0],
    #       [0, -1, 0, 0, 1]
    #     ]
    #   ),
    #   np.array(
    #     [6, 28, 5]
    #   )
    # )

    # 1.2.20
    # x, J_b, _ = simplex_first_phase(
    #   np.array(
    #     [
    #       [-1, 3, 1, 0, 0],
    #       [1, 1, 0, 1, 0],
    #       [0, -1, 0, 0, 1]
    #     ]
    #   ),
    #   np.array(
    #     [3, 5, -3]
    #   )
    # )

    # 1.2.7
    x, J_b, _ = simplex_first_phase(
      np.array(
        [
          [1, 1, 1, 0],
          [1, -1, 0, -1]
        ]
      ),
      np.array(
        [2, 3]
      )
    )

    print(f"x*: {x}, J_b*: {J_b}\n{_}")
