import os
from datasets.base_dataset import Dataset


class MusDBMusicDataset(Dataset):
    """
    musdb18hq style music dataset.

    Expected layout:
        root/
            test/
                song/
                    *bass.wav
                    *drums.wav
                    *mixture.wav
                    *other.wav
                    *vocals.wav
            train/
                song/
                    *bass.wav
                    *drums.wav
                    *mixture.wav
                    *other.wav
                    *vocals.wav
                ...

    Sampling returns uniformly across instrument types:
        (audio_path, class_name)
    """

    def _extract_metadata(self, path: str, fname: str) -> str:
        """
        Extract instrument/component name from filename.

        Args:
            path: Full path to audio file
            fname: Filename

        Returns:
            class_name (filename without extension)
        """
        return os.path.splitext(fname)[0]

if __name__ == "__main__":
    # Point to a test directory with audio files
    dataset = MusDBMusicDataset("F:/datasets/musdb18hq")
    
    print(f"Dataset size: {len(dataset)}")
    
    # Sample a few files
    for _ in range(3):
        path, class_name = dataset.sample()
        print(f"  {class_name}: {path}")