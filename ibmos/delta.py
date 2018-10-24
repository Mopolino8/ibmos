"""Delta functions."""

import numpy as np

romaNumPoints = 2


def roma(r, dr):
    """Delta function from Roma et al. JCP (3-point wide)"""
    d = np.zeros_like(r)
    absr = np.abs(r / dr)
    m3 = absr > 1.5
    m1 = absr <= 0.5
    d[m1] = (1 + np.sqrt(1 - 3 * absr[m1] ** 2)) / (3 * dr)
    m2 = np.logical_not(np.logical_or(m1, m3))
    d[m2] = (5 - 3 * absr[m2] - np.sqrt(1 - 3 * (1 - absr[m2]) ** 2)) / (6 * dr)
    return d


gaussNumPoints = 15


def gauss(r, dr):
    """Gaussian delta function from 10.1016/j.jcp.2016.06.014"""
    d = np.zeros_like(r)
    absr = np.abs(r / dr)
    m1 = absr <= 14
    d[m1] = (np.pi / (36 * dr ** 2)) ** 0.5 * np.exp(-np.pi ** 2 * absr[m1] ** 2 / 36)
    return d


bao6NumPoints = 4


def bao6(r, dr):
    """Bao's delta function from 10.1016/j.jcp.2016.04.024"""
    K = 59 / 60 - np.sqrt(29) / 20

    d, r = np.zeros_like(r), r / dr

    β = lambda x: 9 / 4 - 3 / 2 * (K + x ** 2) + (22 / 3 - 7 * K) * x - 7 / 3 * x ** 3
    γ = lambda x: -11 / 32 * x ** 2 + \
                  3 / 32 * (2 * K + x ** 2) * x ** 2 + \
                  1 / 72 * ((3 * K - 1) * x + x ** 3) ** 2 + \
                  1 / 18 * ((4 - 3 * K) * x - x ** 3) ** 2

    ϕm3 = lambda x: (-β(x) + np.sign(3 / 2 - K) * np.sqrt(β(x) ** 2 - 112 * γ(x))) / 56
    ϕm2 = lambda x: -3 * ϕm3(x) - 1 / 16 + (K + x ** 2) / 8 + (3 * K - 1) * x / 12 + x ** 3 / 12
    ϕm1 = lambda x: 2 * ϕm3(x) + 1 / 4 + (4 - 3 * K) * x / 6 - x ** 3 / 6
    ϕp0 = lambda x: 2 * ϕm3(x) + 5 / 8 - (K + x ** 2) / 4
    ϕp1 = lambda x: -3 * ϕm3(x) + 1 / 4 - (4 - 3 * K) * x / 6 + x ** 3 / 6
    ϕp2 = lambda x: ϕm3(x) - 1 / 16 + (K + x ** 2) / 8 - (3 * K - 1) * x / 12 - x ** 3 / 12

    rm3 = (0 <= (r + 3)) * ((r + 3) < 1)
    rm2 = (0 <= (r + 2)) * ((r + 2) < 1)
    rm1 = (0 <= (r + 1)) * ((r + 1) < 1)
    rp0 = (0 <= (r - 0)) * ((r - 0) < 1)
    rp1 = (0 <= (r - 1)) * ((r - 1) < 1)
    rp2 = (0 <= (r - 2)) * ((r - 2) < 1)

    d[rm3] = ϕm3(r[rm3] + 3)
    d[rm2] = ϕm2(r[rm2] + 2)
    d[rm1] = ϕm1(r[rm1] + 1)
    d[rp0] = ϕp0(r[rp0] - 0)
    d[rp1] = ϕp1(r[rp1] - 1)
    d[rp2] = ϕp2(r[rp2] - 2)

    return d / dr


defaultNumPoints = romaNumPoints
defaultFunction = roma
