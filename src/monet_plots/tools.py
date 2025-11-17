# src/monet_plots/tools.py
import numpy as np

def wsdir2uv(ws, wdir):
    """Converts wind speed and direction to u and v components.

    Args:
        ws (numpy.ndarray): The wind speed.
        wdir (numpy.ndarray): The wind direction.

    Returns:
        tuple: A tuple containing the u and v components of the wind.
    """
    rad = np.pi / 180.
    u = -ws * np.sin(wdir * rad)
    v = -ws * np.cos(wdir * rad)
    return u, v
