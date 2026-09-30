import os
from datasets.base_dataset import Dataset

class NigensDataset(Dataset):
    """
    Nigens style dataset.

    Expected layout:
        root/
            train/
                class_name/
                    audio.wav
            test/
                class_name/
                    audio.wav

    Sampling returns:
        (audio_path, class_name)
    """

    def __init__(self, root_dir: str, exts=(".wav",), split: str = None, exclude_classes: list = None):
        super().__init__(root_dir, exts, split, exclude_classes)

    def _extract_metadata(self, path: str, fname: str) -> str:
        # .../train/alarm/audio.wav
        #           ^^^^
        #        class name
        return os.path.basename((os.path.dirname(path)))
    

class FSD50KFMADataset(Dataset):
    """
    FSD50K-FMA style dataset.

    Expected layout:
        root/
            class_name/
                train/
                    subclass/
                        audio.wav
                test/
                    subclass/
                        audio.wav

    Sampling returns:
        (audio_path, class_name)
    """

    def __init__(self, root_dir: str, exts=(".wav",), split: str = None, exclude_classes: list = None):
        super().__init__(root_dir, exts, split, exclude_classes)

    def _extract_metadata(self, path: str, fname: str) -> str:
        # .../bell/train/Bicycle_bell/audio.wav
        #     ^^^^
        #   class name
        return os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(path))))


class FreesoundDataset(Dataset):
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