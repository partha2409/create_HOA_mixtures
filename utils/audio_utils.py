import random
import numpy as np
import soundfile as sf
import librosa



def load_audio(path, target_fs=24000, start_sample=None, num_samples=None):
    try:
        if start_sample is not None:
            info = sf.info(path)
            if info.samplerate != target_fs:
                start_orig = int(start_sample * info.samplerate / target_fs)
                num_orig = int(num_samples * info.samplerate / target_fs)
            else:
                start_orig = start_sample
                num_orig = num_samples

            x, fs = sf.read(path, start=start_orig, stop=start_orig + num_orig)
        else:
            x, fs = sf.read(path)

    except Exception:
        return None

    if x.ndim > 1:
        x = x.mean(axis=1)

    if fs != target_fs:
        x = librosa.resample(y=x, orig_sr=fs, target_sr=target_fs)

    return x.astype(np.float32)


def load_audio_segment(path, seg_len_samples, target_fs=24000, rms_thresh=1e-4, max_tries=2000):
    try:
        info = sf.info(path)

        total_samples = int(info.frames * target_fs / info.samplerate)

        # File shorter than desired segment
        if total_samples <= seg_len_samples:
            x = load_audio(path, target_fs)
            return (x, 0) if x is not None else (None, None)

        max_start = total_samples - seg_len_samples

        for _ in range(max_tries):
            start_sample = random.randint(0, max_start)
            x = load_audio(path, target_fs=target_fs, start_sample=start_sample, num_samples=seg_len_samples)
            if x is None or len(x) == 0:
                continue
            rms = np.sqrt(np.mean(x ** 2))
            if rms > rms_thresh:
                return x, start_sample

        return None, None

    except Exception as e:
        print(f"Error loading segment from {path}: {e}")
        return None, None


def load_background_segment(path, seg_len_samples, target_fs=24000, allow_random_segment=True):
    """
    Load a segment from a background audio file.
    """

    try:
        info = sf.info(path)

        total_samples = int(info.frames * target_fs / info.samplerate)

        # File is shorter than requested segment
        if total_samples <= seg_len_samples:
            return load_audio(path, target_fs)

        # Choose segment start
        if allow_random_segment and total_samples > seg_len_samples * 2:
            start_sample = random.randint(0, total_samples - seg_len_samples)
        else:
            start_sample = 0

        return load_audio(path, target_fs=target_fs, start_sample=start_sample, num_samples=seg_len_samples) 

    except Exception as e:
        print(f"Error loading background segment from {path}: {e}")
        return None
    

#TODO: use a more robust method ?
def extract_non_silent_segment(x, seg_len, rms_thresh=1e-4, max_tries=2000):
    """
    Extract a non-silent segment from audio x of length seg_len.
    If x is shorter than seg_len, return the whole audio.

    Returns:
        segment (numpy array), start_index (int)
    """
    # If audio is shorter than desired segment, return whole audio
    if len(x) <= seg_len:
        return x, 0

    # Try to find a segment with RMS above threshold
    for _ in range(max_tries):
        s = random.randint(0, len(x) - seg_len)
        seg = x[s:s + seg_len]
        if np.sqrt(np.mean(seg**2)) > rms_thresh:
            return seg, s

    # If no high-RMS segment found, just return a random segment anyway
    s = random.randint(0, len(x) - seg_len)
    return x[s:s + seg_len], s