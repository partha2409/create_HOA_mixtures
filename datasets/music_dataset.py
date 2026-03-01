import os
import random
from typing import Tuple, List


class MusDBMusicDataset:
    """
    musdb18hq style music-dataset.

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

    Sampling returns:
        (audio_path, class_name)
    """

    def __init__(self, root_dir: str, exts=(".wav",)):
        self.root_dir = root_dir
        self.exts = exts

        self._index = self._build_index()

        if len(self._index) == 0:
            raise RuntimeError(f"No audio files found in {root_dir}")

    def _build_index(self) -> List[Tuple[str, str]]:
        index = []

        for root, _, files in os.walk(self.root_dir):
            for fname in files:
                if fname.lower().endswith(self.exts):
                    path = os.path.join(root, fname)
                    class_name = os.path.splitext(fname)[0]
                    index.append((path, class_name))

        return index

    def sample(self) -> Tuple[str, str]:
        """
        Sample a random Speech file.

        Returns:
            (audio_path, class_name)
        """
        return random.choice(self._index)

    def __len__(self):
        return len(self._index)

if __name__ == "__main__":
    # Point to a test directory with audio files
    dataset = MusDBMusicDataset("F:/datasets/musdb18hq")
    
    print(f"Dataset size: {len(dataset)}")
    
    # Sample a few files
    for _ in range(3):
        path, class_name = dataset.sample()
        print(f"  {class_name}: {path}")