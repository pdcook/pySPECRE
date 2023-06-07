import matplotlib.pyplot as plt
import numpy as np

import pySPECRE


# test problem from source paper
def B(c: complex) -> np.ndarray:
    """
    Consider the 5 × 5 matrix whose non-zero elements are given by
    B(1, 1) = c(1 + i) + c^2
    B(2, 2) = c(− cos t1 + i sin t1) + c^2,
    B(3, 3) = c(− cos t2 + i sin t2) + c^3,
    B(4, 4) = c(cos t3 − i sin t3) + c^4,
    B(5, 5) = c(cos t4 + i sin t4) + c^5,
    and B(2, 1) = B(3, 2) = B(4, 3) = B(5, 4) = 0.1

    t1 = 3 * np.pi / 20
    t2 = 11 * np.pi / 20
    t3 = 7 * np.pi / 20
    t4 = np.pi / 20

    by definition this has a quintuple degenerate eigenvalue at c = 0

    """

    t1 = 3 * np.pi / 20
    t2 = 11 * np.pi / 20
    t3 = 7 * np.pi / 20
    t4 = np.pi / 20

    return np.array(
        [
            [c * (1 + 1j) + c ** 2, 0, 0, 0, 0],
            [0.1, c * (-np.cos(t1) + 1j * np.sin(t1)) + c ** 2, 0, 0, 0],
            [0, 0.1, c * (-np.cos(t2) + 1j * np.sin(t2)) + c ** 3, 0, 0],
            [0, 0, 0.1, c * (np.cos(t3) - 1j * np.sin(t3)) + c ** 4, 0],
            [0, 0, 0, 0.1, c * (np.cos(t4) + 1j * np.sin(t4)) + c ** 5],
        ]
    )


def quintuple():

    C = np.arange(-1, 1, 1e-3)

    C, ws, vs = pySPECRE.pySPECRE.SPECRE(B, C)

    for i in range(5):
        l = plt.plot(
            np.real(ws[0, i]),
            np.imag(ws[0, i]),
            ls=None,
            marker="o",
            fillstyle="none",
            markersize=10,
        )
        plt.plot(np.real(ws[:, i]), np.imag(ws[:, i]), lw=3, color=l[0].get_color())

    plt.show()


if __name__ == "__main__":
    quintuple()
