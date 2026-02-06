import random
import numpy as np


def cartesian_to_az_el_dist(src_pos, mic_pos):
    """
    Convert Cartesian source position to azimuth, elevation, and distance.

    Returns:
        azimuth (deg), elevation (deg), distance (cm)
    """
    src = np.asarray(src_pos, dtype=float)
    mic = np.asarray(mic_pos, dtype=float)

    dx, dy, dz = src - mic

    dist = np.sqrt(dx**2 + dy**2 + dz**2)
    dist_cm = dist * 100.0

    az = np.degrees(np.arctan2(dy, dx))
    el = np.degrees(np.arctan2(dz, np.sqrt(dx**2 + dy**2)))

    return az, el, dist_cm


def az_el_to_unit_vec(az_deg, el_deg):
    """
    Convert azimuth/elevation (deg) to 3D unit vector.
    """
    az = np.radians(az_deg)
    el = np.radians(el_deg)

    return np.array([
        np.cos(el) * np.cos(az),
        np.cos(el) * np.sin(az),
        np.sin(el)
    ])


def angular_distance_deg(az1, el1, az2, el2):
    """
    Angular distance (deg) between two azimuth/elevation directions.
    """
    u1 = az_el_to_unit_vec(az1, el1)
    u2 = az_el_to_unit_vec(az2, el2)

    dot = np.clip(np.dot(u1, u2), -1.0, 1.0)
    return np.degrees(np.arccos(dot))




def doa_unit_vector(src_pos, mic_pos):
    """
    Unit DOA vector pointing from mic to source.
    """
    v = np.asarray(src_pos, dtype=float) - np.asarray(mic_pos, dtype=float)
    norm = np.linalg.norm(v)

    if norm < 1e-8:
        return [0.0, 0.0, 0.0]

    return (v / norm).tolist()