import os
import yaml


from datasets.speech_dataset import VctkSpeechDataset
from datasets.music_dataset import MusDBMusicDataset
from datasets.general_sounds_dataset import NigensDataset, FSD50KFMADataset, FreesoundDataset

from datasets.background_dataset import BackgroundDataset
from renderer.simulate_per_room import simulate_per_room



def main(config):
    """
    Main pipeline to generate synthetic room mixtures.

    Args:
        config: dict loaded from YAML
    """
    
    # Prepare datasets dict
    datasets = {}
    modes = [m.lower() for m in config.get("dataset_mode", ["speech", "music", "fsd50k_fma"])]

    if "fsd50k_fma" in modes and "fsd50k_fma_dir" in config:
        datasets["fsd50k_fma"] = FSD50KFMADataset(config["fsd50k_fma_dir"], split=config.get("split"), exclude_classes=config.get("fsd50k_fma_exclude_classes", []))

    if "vctk" in modes and "vctk_dir" in config:
        datasets["vctk"] = VctkSpeechDataset(config["vctk_dir"])

    if "music" in modes and "music_dir" in config:
        datasets["music"] = MusDBMusicDataset(config["music_dir"])

    if "nigens" in modes and "nigens_dir" in config:
        datasets["nigens"] = NigensDataset(config["nigens_dir"], split=config.get("split"), exclude_classes=config.get("nigens_exclude_classes", []))

    if "freesound" in modes and "freesound_dir" in config:
        datasets["freesound"] = FreesoundDataset(config["freesound_dir"])

    if config['n_diffuse_sources'] > 0 and config.get("background_dir", None) is not None:
        datasets["background"] = BackgroundDataset(config["background_dir"])
    
    if len(datasets) == 0:
        raise RuntimeError("No datasets initialized. Check your config and dataset_mode.")

    # Make sure output folder exists
    os.makedirs(config["out_dir"], exist_ok=True)

    # Generate mixtures for all rooms
    for room_idx in range(config["n_rooms"]):
        if os.path.exists(os.path.join(config["out_dir"], "room_"+str(room_idx))) == False:
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
