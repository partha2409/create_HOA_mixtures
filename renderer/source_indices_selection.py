import random
from utils.geometry_utils import cartesian_to_az_el_dist, angular_distance_deg


def sample_spatial_sources(src_positions, mic_pos, K, min_sep_deg, max_tries=2000):
    """
    Sample K source indices with minimum angular separation.
    Guarantees K sources even if spatial constraint is too strict.

    Returns:
        list of K source indices
    """
    chosen = []
    chosen_dirs = []

    tries = 0
    while len(chosen) < K and tries < max_tries:
        idx = random.randrange(len(src_positions))
        if idx in chosen:
            tries += 1
            continue

        az, el, _ = cartesian_to_az_el_dist(src_positions[idx], mic_pos)
        if all(angular_distance_deg(az, el, az0, el0) >= min_sep_deg for az0, el0 in chosen_dirs):
            chosen.append(idx)
            chosen_dirs.append((az, el))
        tries += 1

    # If we didn’t reach K, fill the remaining randomly ignoring angular separation
    while len(chosen) < K:
        print("Warning: could not sample enough spatially separated sources; relaxing constraint.")
        remaining = [i for i in range(len(src_positions)) if i not in chosen]
        if not remaining:
            break
        idx = random.choice(remaining)
        chosen.append(idx)

    return chosen