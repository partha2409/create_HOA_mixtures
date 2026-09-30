import os
import random
from typing import Tuple, List
from abc import ABC, abstractmethod


class Dataset(ABC):
    """
    Base class for audio datasets.
    
    Subclasses must implement `_extract_metadata()` to define how metadata
    is extracted from file paths.
    """

    def __init__(self, root_dir: str, exts: Tuple[str, ...] = (".wav",".flac"), split: str = None, exclude_classes: List[str] = None):
        """
        Initialize dataset.

        Args:
            root_dir: Root directory containing audio files
            exts: Tuple of file extensions to match (e.g., (".wav", ".flac"))
            split: Optional split name (e.g., "train", "test") to filter files
            exclude_classes: Optional list of class names to exclude from the dataset 
        """
        self.root_dir = root_dir
        self.exts = exts
        self.split = split
        self.exclude_classes = exclude_classes if exclude_classes is not None else []

        self._index = self._build_index()

        if len(self._index) == 0:
            raise RuntimeError(f"No audio files found in {root_dir}")

    def _build_index(self) -> List[Tuple[str, str]]:
        """
        Build index of all audio files in root directory.

        Returns:
            List of (audio_path, metadata) tuples
        """
        index = []

        for root, _, files in os.walk(self.root_dir):
            
            # Only include requested split
            if self.split is not None:
                directories = [x.lower() for x in root.split(os.sep)]

                if self.split.lower() not in directories:
                    continue

            for fname in files:
                if fname.lower().endswith(self.exts):
                    path = os.path.join(root, fname)
                    metadata = self._extract_metadata(path, fname)

                    if metadata in self.exclude_classes:
                        continue
                    index.append((path, metadata))

        return index

    @abstractmethod
    def _extract_metadata(self, path: str, fname: str) -> str:
        """
        Extract metadata from a file path.

        Args:
            path: Full path to the audio file
            fname: Filename

        Returns:
            Metadata string (class name, speaker ID, etc.)
        """
        pass

    def sample(self) -> Tuple[str, str]:
        """
        Sample a random audio file uniformly across all metadata categories.

        Returns:
            (audio_path, metadata)
        """
        # Group index by metadata
        metadata_groups = {}
        for path, metadata in self._index:
            if metadata not in metadata_groups:
                metadata_groups[metadata] = []
            metadata_groups[metadata].append((path, metadata))

        # Uniformly sample a metadata group, then uniformly sample an item
        group_items = random.choice(list(metadata_groups.values()))
        return random.choice(group_items)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self._index)
