import os
from datasets.base_dataset import Dataset


class VctkSpeechDataset(Dataset):
    """
    VCTK-Corpus-0.92-style speech dataset.

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

    Sampling returns uniformly across speakers:
        (audio_path, speaker_id)
    """

    def __init__(self, root_dir: str, exts=("mic1.flac",)):
        """
        Initialize speech dataset.

        Args:
            root_dir: Root directory of the dataset
            exts: File extension to sample from (default: mic1.flac only)
        """
        super().__init__(root_dir, exts)

    def _extract_metadata(self, path: str, fname: str) -> str:
        """
        Extract speaker ID from directory structure.

        Args:
            path: Full path to audio file
            fname: Filename

        Returns:
            speaker_id (directory name containing the file)
        """
        return os.path.basename(os.path.dirname(path))

if __name__ == "__main__":
    # Point to a test directory with audio files
    dataset = VctkSpeechDataset("F:/datasets/VCTK-Corpus-0.92")
    
    print(f"Dataset size: {len(dataset)}")
    
    # Sample a few files
    for _ in range(3):
        path, class_name = dataset.sample()
        print(f"  {class_name}: {path}")