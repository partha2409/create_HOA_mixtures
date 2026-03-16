import os
from datasets.base_dataset import Dataset


class FreesoundSFXDataset(Dataset):
    """
    Freesound-style SFX dataset.

    Expected layout:
        root/
            class_1/
                *.wav
            class_2/
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
        return os.path.basename(os.path.dirname(path))
