import numpy as np
from typing import Callable

def rk4(f: Callable[[float, np.ndarray], np.ndarray], y0: np.ndarray, t0: float, h: float) -> np.ndarray:
    """
    Solves a single step of an ODE using the 4th Order Runge-Kutta method.

    Args:
        f: The derivative function, f(t, y).
        y0: The initial state vector at time t0.
        t0: The initial time.
        h: The time step.

    Returns:
        The new state vector at time t0 + h.
    """
    k1 = h * f(t0, y0)
    k2 = h * f(t0 + 0.5 * h, y0 + 0.5 * k1)
    k3 = h * f(t0 + 0.5 * h, y0 + 0.5 * k2)
    k4 = h * f(t0 + h, y0 + k3)
    
    y1 = y0 + (k1 + 2*k2 + 2*k3 + k4) / 6.0
    return y1