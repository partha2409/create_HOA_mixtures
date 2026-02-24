import os
import json
import numpy as np
import soundfile as sf
from renderer.render_mixture import render_mixture
from utils.common_utils import round_floats


def simulate_per_room(room_idx, rir_dir, out_dir, datasets, config):
    """
    Generate all mixtures for a single room.

    Args:
        room_idx: int
        rir_dir: str
        out_dir: str
        datasets: dict of available dataset instances
        config: dict

    Returns:
        list of (wav_path, json_path)
    """
    room_out_dir = os.path.join(out_dir, f"room_{room_idx}")
    os.makedirs(room_out_dir, exist_ok=True)

    # Load room data once
    rir_path = os.path.join(rir_dir, f"room_{room_idx}", "rirs.npy")
    metadata_path = os.path.join(rir_dir, f"room_{room_idx}", "metadata.json")

    rirs = np.load(rir_path)
    with open(metadata_path, "r") as f:
        room_meta = json.load(f)

    hoa_order = config["hoa_order"]
    n_ch = (hoa_order + 1) ** 2

    rirs = rirs[:, :n_ch, :]

    output_info = []

    for mix_idx in range(config["mixtures_per_room"]):

        hoa, events = render_mixture(rirs=rirs, room_meta=room_meta, datasets=datasets, config=config)

        # Normalize
        peak = np.max(np.abs(hoa))
        if peak > 0:
            hoa /= peak * 1.05

        # Save audio
        wav_path = os.path.join(room_out_dir, f"mix_{mix_idx:03d}.wav")
        sf.write(wav_path, hoa.T, config["fs"])

        # Save metadata
        meta = round_floats({
            "room_id": room_idx,
            "fs": config["fs"],
            "duration_sec": config["scene_len_sec"],
            "num_events": len(events),
            "events": events,
            "hoa_order": hoa_order,
            "ambisonics_normalization_scheme": "SN3D",
        })

        json_path = wav_path.replace(".wav", ".json")
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=4)

        output_info.append((wav_path, json_path))
        print(f"Room {room_idx} | Mix {mix_idx} done with {len(events)} events.")

    return output_info
