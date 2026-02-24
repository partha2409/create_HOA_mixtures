import os
import yaml

from datasets.freesound_dataset import FreesoundSFXDataset
from datasets.speech_dataset import VctkSpeechDataset
# from datasets.music import MusicDataset
from renderer.simulate_per_room import simulate_per_room


def main(config):
    """
    Main pipeline to generate synthetic room mixtures.

    Args:
        config: dict loaded from YAML
    """
    # Prepare datasets dict
    datasets = {}
    mode = config.get("dataset_mode", "all").lower()

    if mode in ["sfx", "all"] and "sfx_dir" in config:
        datasets["sfx"] = FreesoundSFXDataset(config["sfx_dir"])
    if mode in ["speech", "all"] and "speech_dir" in config:
        pass
        datasets["speech"] = VctkSpeechDataset(config["speech_dir"])
    if mode in ["music", "all"] and "music_dir" in config:
        pass
        #datasets["music"] = MusicDataset(config["music_dir"])

    if len(datasets) == 0:
        raise RuntimeError("No datasets initialized. Check your config and dataset_mode.")

    # Make sure output folder exists
    os.makedirs(config["out_dir"], exist_ok=True)

    # Generate mixtures for all rooms
    for room_idx in range(config["n_rooms"]):
        simulate_per_room(room_idx, config["rir_dir"], config["out_dir"], datasets, config)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Synthetic Audio Dataset Generator")
    parser.add_argument( "--config", type=str, default=os.path.join("configs", "config.yaml"), help="Path to the YAML configuration file")
    args = parser.parse_args()

    # Load configuration and run main
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    main(config)
