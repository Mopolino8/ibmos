import numpy as np
from scipy.interpolate import interp1d

from .delta import defaultFunction, defaultNumPoints
from .solid import Solid


def cylinder(name, x, y, r, ds, δ=defaultFunction, n=defaultNumPoints):
    l = int(2 * np.pi * r / ds) + 1
    θ = 2 * np.pi * np.r_[:l] / l
    solid = Solid(name, x + r * np.cos(θ), y + r * np.sin(θ), 2 * np.pi * r / l * np.ones(l), δ, n)

    return solid


def superellipse(name, x, y, a, b, m, ds, δ=defaultFunction, n=defaultNumPoints):
    t = np.linspace(0, 2 * np.pi, int(4 * np.pi * (a + b) / ds))
    _ξ = x + a * np.abs(np.cos(t)) ** (2 / m) * np.sign(np.cos(t))
    _η = y + b * np.abs(np.sin(t)) ** (2 / m) * np.sign(np.sin(t))
    _s = np.r_[0, np.cumsum(np.sqrt(np.diff(_ξ) ** 2 + np.diff(_η) ** 2))]

    l = int(_s[-1] / ds) + 1
    s = _s[-1] * np.r_[:l] / l
    ξ, η = (interp1d(_s, f, kind='cubic')(s) for f in (_ξ, _η))

    solid = Solid(name, ξ, η, _s[-1] * np.ones(l) / l, δ, n)

    return solid
