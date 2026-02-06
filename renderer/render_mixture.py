import os
import random
import numpy as np
from scipy.signal import fftconvolve
from utils.audio_utils import load_audio, extract_non_silent_segment
from utils.geometry_utils import cartesian_to_az_el_dist, doa_unit_vector
from renderer.source_indices_selection import sample_spatial_sources
from renderer.overlap_events import can_place, mark_timeline


def render_mixture(rirs, room_meta, datasets, config):
    """
    Render a single acoustic mixture for one room.

    Args:
        rirs: np.ndarray [n_src, n_ch, rir_len]
        room_meta: dict loaded from room metadata.json
        datasets: dict of available dataset instances
        config: dict

    Returns:
        hoa: np.ndarray [n_ch, scene_len]
        events: list of dicts
    """
    mic_pos = room_meta["mic"]["mic_position"]
    src_positions = room_meta["sources"]["stationary"]["stationary_source_positions"]

    n_ch = rirs.shape[1]
    rir_len = rirs.shape[2]

    hoa = np.zeros(
        (n_ch, config["scene_len"] + rir_len - 1),
        dtype=np.float32,
    )
    timeline = np.zeros(config["scene_len"], dtype=np.int32)

    # Number of sources
    K = random.randint(config["min_sources"], config["max_sources"])
    print(f"Placing {K} sources")

    src_idxs = sample_spatial_sources(
        src_positions,
        mic_pos,
        K,
        config["min_angular_sep_deg"],
    )

    events = []

    dataset_keys = list(datasets.keys())

    for src_idx in src_idxs:
        # -----------------------------
        # Sample dataset + audio
        # -----------------------------
        dataset_name = random.choice(dataset_keys)
        dataset = datasets[dataset_name]
        audio_path, audio_event = dataset.sample()

        x = load_audio(audio_path, target_fs=config["fs"])

        dur = int(
            random.uniform(
                config["min_event_dur_sec"],
                config["max_event_dur_sec"],
            ) * config["fs"]
        )

        seg = extract_non_silent_segment(
            x,
            dur,
            rms_thresh=config["rms_thresh"],
            max_tries=config["max_tries"],
        )

        # -----------------------------
        # Temporal placement
        # -----------------------------
        placed = False
        for _ in range(config["max_tries"]):
            start = random.randint(0, config["scene_len"] - len(seg))
            end = start + len(seg)
            if can_place(start, end, timeline, config["max_overlap"]):
                placed = True
                break

        if not placed:
            start = random.randint(0, config["scene_len"] - len(seg))
            end = start + len(seg)

        mark_timeline(start, end, timeline)

        # -----------------------------
        # Convolution
        # -----------------------------
        for ch in range(n_ch):
            hoa[ch, start : start + len(seg) + rir_len - 1] += fftconvolve(
                seg, rirs[src_idx, ch], mode="full"
            )

        # -----------------------------
        # Metadata
        # -----------------------------
        az, el, dist = cartesian_to_az_el_dist(
            src_positions[src_idx], mic_pos
        )
        doa_vec = doa_unit_vector(src_positions[src_idx], mic_pos)

        events.append(
            {
                "audio_file": os.path.basename(audio_path),
                "audio_event": audio_event,
                "dataset": dataset_name,
                "source_index": src_idx,
                "azimuth_deg": az,
                "elevation_deg": el,
                "distance_cm": dist,
                "doa_unit_vector": doa_vec,
                "start_time_sec": start / config["fs"],
                "end_time_sec": end / config["fs"],
            }
        )

    return hoa[:, : config["scene_len"]], events
