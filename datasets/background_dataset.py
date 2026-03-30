import os
from datasets.base_dataset import Dataset


class BackgroundDataset(Dataset):
    """
    Background dataset used for diffuse sounds.

    Expected layout:
        root/
                *.wav
                *.wav

    Sampling returns uniformly across classes:
        (audio_path, class_name)
    """

    def _extract_metadata(self, path: str, fname: str) -> str:
        """
        Extract class name from directory structure.

        Args:
            path: Full path to audio file
            fname: Filename

        Returns:
            class_name (directory name containing the file)
        """
        return fname
