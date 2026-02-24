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
        (audio_path, class_name)
    """

    def __init__(self, root_dir: str, exts=(".flac",)):
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
                    class_name = os.path.basename(os.path.dirname(path))
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
