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


def load_audio_segment(path, seg_len_samples, target_fs=24000, rms_thresh=1e-4, max_tries=2000):
    """
    Load an event-length non-silent segment without loading the entire file.

    Args:
        path: Path to audio file
        seg_len_samples: Length of segment to extract (in samples at target_fs)
        target_fs: Target sampling rate
        rms_thresh: RMS threshold for considering audio non-silent
        max_tries: Maximum number of random positions to try

    Returns:
        segment (numpy array), start_sample (int) or (None, None) if no suitable segment found
    """
    try:
        info = sf.info(path)
        total_samples = int(info.duration * target_fs) if info.samplerate != target_fs else info.frames

        # If file is shorter than desired segment, load the whole thing
        if total_samples <= seg_len_samples:
            return load_audio(path, target_fs), 0

        for _ in range(max_tries):
            # Choose a random starting position
            max_start = total_samples - seg_len_samples
            start_sample_target = random.randint(0, max_start)

            # Convert to original file's sampling rate for reading if needed
            if info.samplerate != target_fs:
                start_sample_orig = int(start_sample_target * info.samplerate / target_fs)
                seg_len_samples_orig = int(seg_len_samples * info.samplerate / target_fs)
            else:
                start_sample_orig = start_sample_target
                seg_len_samples_orig = seg_len_samples
            x, fs = sf.read(path, start=start_sample_orig, stop=start_sample_orig + seg_len_samples_orig)
            # Check RMS of the portion
            rms = np.sqrt(np.mean(x**2))
            if rms > rms_thresh:
                if x.ndim > 1:
                    x = x.mean(axis=1)
                if fs != target_fs:
                    x = librosa.resample(y=x, orig_sr=fs, target_sr=target_fs)
                return x.astype(np.float32), start_sample_target

        return None, None  # No suitable segment found

    except Exception as e:
        print(f"Error loading segment from {path}: {e}")
        return None, None


def load_background_segment(path, seg_len_samples, target_fs=24000, allow_random_segment=True):
    """
    Load a segment from a background audio file, optimized for long files.
    For long files, loads the first seg_len_samples unless allow_random_segment is True,
    in which case it may load a random segment to add variety.

    Args:
        path: Path to audio file
        seg_len_samples: Length of segment to extract (in samples at target_fs)
        target_fs: Target sampling rate
        allow_random_segment: If True, may load random segments from long files for variety

    Returns:
        segment (numpy array) or None if loading fails
    """
    try:
        info = sf.info(path)
        total_samples = int(info.duration * target_fs) if info.samplerate != target_fs else info.frames

        if total_samples <= seg_len_samples:
            # File is short, load it all
            return load_audio(path, target_fs)

        if allow_random_segment and total_samples > seg_len_samples * 2:
            # Long file, load a random segment for variety
            start_sample_target = random.randint(0, total_samples - seg_len_samples)
        else:
            # Load from the beginning
            start_sample_target = 0

        # Convert to original sampling rate if needed
        if info.samplerate != target_fs:
            start_sample_orig = int(start_sample_target * info.samplerate / target_fs)
            seg_len_samples_orig = int(seg_len_samples * info.samplerate / target_fs)
        else:
            start_sample_orig = start_sample_target
            seg_len_samples_orig = seg_len_samples
        x, fs = sf.read(path, start=start_sample_orig, stop=start_sample_orig + seg_len_samples_orig)
        # Process
        if x.ndim > 1:
            x = x.mean(axis=1)
        if info.samplerate != target_fs:
            x = librosa.resample(y=x, orig_sr=info.samplerate, target_sr=target_fs)

        return x.astype(np.float32)

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