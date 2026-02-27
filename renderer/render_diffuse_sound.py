import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve
from utils.audio_utils import load_audio


def render_diffuse_sound(rirs, datasets, config):
    """
    Render diffuse background using late reverberation only.

    Diffuse field simulation:
        mono background → late HOA RIR → sum multiple sources

    Returns:
        diffuse: np.ndarray [n_ch, scene_len]
        diffuse_meta: list of metadata dicts
    """

    n_positions, n_ch, rir_len = rirs.shape
    scene_len = config["scene_len"]
    fs = config["fs"]
    k = config["n_diffuse_sources"]
    mixing_time_sec = config.get("mixing_time_sec", 0.05)

    diffuse = np.zeros((n_ch, scene_len), dtype=np.float32)
    diffuse_meta = []

    bg_dataset = datasets["background"]

    # --------------------------------------------------------
    # Choose ONE RIR (late reverb statistics are position-independent)
    # --------------------------------------------------------
    pos_idx = np.random.randint(0, n_positions)
    rir_full = rirs[pos_idx]  # [n_ch, rir_len]

    # --------------------------------------------------------
    # Remove early reflections (keep only late reverberation)
    # --------------------------------------------------------
    mixing_samples = int(mixing_time_sec * fs)
    
    # Crop early part
    rir_late = np.zeros_like(rir_full)
    if mixing_samples < rir_len:
        rir_late[:, :] = rir_full[:, mixing_samples:]
        late_reverb_only = True
    else:
        # RIR too short, keep all. This is a fallback to avoid empty RIRs, but may include some early reflections if the RIR is very short.
        rir_late[:, :] = rir_full
        late_reverb_only = False

    # --------------------------------------------------------
    # Generate k diffuse sources
    # --------------------------------------------------------
    for _ in range(k):

        # Sample mono background clip
        bg_path, _ = bg_dataset.sample()
        clip = load_audio(bg_path, target_fs=fs)

        if clip.ndim > 1:
            clip = clip.mean(axis=1)

        if len(clip) >= scene_len:
            clip = clip[:scene_len]
        else:
            clip = np.pad(clip, (0, scene_len - len(clip)))

        # Convolve mono signal with late HOA RIR
        for ch in range(n_ch):
            convolved = fftconvolve(clip, rir_late[ch], mode="full")[:scene_len]

            diffuse[ch] += convolved.astype(np.float32)

        diffuse_meta.append({
            "audio_file": bg_path,
            "rir_position_index": int(pos_idx),
            "late_reverb_only": late_reverb_only,
            "mixing_time_sec": mixing_time_sec
        })

    return diffuse, diffuse_meta