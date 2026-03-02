import os
import random
from typing import Tuple, List


class VctkSpeechDataset:
    """
    VCTK-Corpus-0.92-style SFX dataset.

    Expected layout:
        root/
            audio_dir/
                speaker_1/
                    *_mic1.flac
                    *_mic2.flac
                speaker_2/
                    *_mic1.flac
                    *_mic2.flac
                ...

    Sampling returns:
        (audio_path, speaker_id)
    """

    def __init__(self, root_dir: str, exts=("mic1.flac",)): # Only sample from mic1 by default
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
                    id = os.path.basename(os.path.dirname(path))
                    index.append((path, id))

        return index

    def sample(self) -> Tuple[str, str]:
        """
        Sample a random Speech file.

        Returns:
            (audio_path, speaker_id)
        """
        return random.choice(self._index)

    def __len__(self):
        return len(self._index)

if __name__ == "__main__":
    # Point to a test directory with audio files
    dataset = VctkSpeechDataset("F:/datasets/VCTK-Corpus-0.92")
    
    print(f"Dataset size: {len(dataset)}")
    
    # Sample a few files
    for _ in range(3):
        path, class_name = dataset.sample()
        print(f"  {class_name}: {path}")