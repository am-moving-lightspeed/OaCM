import numpy as nmp
import inverter as inv


if __name__ == '__main__':
    # A = nmp.array(
    #   [
    #     [-24, 20, -5],
    #     [18, -15, 4],
    #     [5, -4, 1]
    #   ],
    #   dtype = nmp.float_
    # )
    # x = nmp.array(
    #   [
    #     [2],
    #     [2],
    #     [2]
    #   ],
    #   dtype = nmp.float_
    # )
    # index = 1

    # A = nmp.array(
    #   [
    #     [1, 0, 3],
    #     [5, 5, 7],
    #     [3, 0, 4]
    #   ],
    #   dtype = nmp.float_
    # )
    # x = nmp.array(
    #   [
    #     [3],
    #     [2],
    #     [4]
    #   ],
    #   dtype = nmp.float_
    # )
    # index = 2

    A = nmp.array(
      [
        [0, 0, 0],
        [0, 0, 7],
        [0, 0, 0]
      ],
      dtype = nmp.float_
    )
    x = nmp.array(
      [
        [3],
        [2],
        [4]
      ],
      dtype = nmp.float_
    )
    index = 2

    print(inv.inv_from_inv(A, x, index))
