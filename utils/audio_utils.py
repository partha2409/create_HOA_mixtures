import random
import numpy as np
import soundfile as sf
import librosa


def load_audio(path, target_fs=24000):
    try:
        x, fs = sf.read(path)
    except Exception:
        return None
    if x.ndim > 1:
        x = x.mean(axis=1)
    if fs != target_fs:
        x = librosa.resample(y=x, orig_sr=fs, target_sr=target_fs)
    return x.astype(np.float32)


#TODO: use a more robust method ?
def extract_non_silent_segment(x, seg_len, rms_thresh=1e-4, max_tries=2000):
    """
    Extract a non-silent segment from audio x of length seg_len.
    If x is shorter than seg_len, return the whole audio.
    
    Returns:
        segment (numpy array)
    """
    # If audio is shorter than desired segment, return whole audio
    if len(x) <= seg_len:
        return x

    # Try to find a segment with RMS above threshold
    for _ in range(max_tries):
        s = random.randint(0, len(x) - seg_len)
        seg = x[s:s + seg_len]
        if np.sqrt(np.mean(seg**2)) > rms_thresh:
            return seg

    # If no high-RMS segment found, just return a random segment anyway
    s = random.randint(0, len(x) - seg_len)
    return x[s:s + seg_len]
