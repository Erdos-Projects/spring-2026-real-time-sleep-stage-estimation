# from pathlib import Path

# config_path = Path(__file__).resolve().parent.parent.parent / 'config'

# print(f"Config path: {config_path} loaded, file-exists: {config_path.is_dir()}")


import json
from pathlib import Path

from omegaconf import OmegaConf


# Load the YAML configuration file in config directory
def load_config(filename):
	return OmegaConf.load(Path(__file__).parent / filename)


# Example of accessing a specific configuration file
dir_config = load_config("dir-config.yaml")
apnea_model_config = load_config("apnea-model.yaml")
sleep_model_config = load_config("sleep-model.yaml")

random_seed = 2542 #255
