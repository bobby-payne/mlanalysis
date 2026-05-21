import yaml
from pathlib import Path


class Config:
    def __init__(self, relative_path):
        # Resolve path relative to this file, not the working directory
        base_dir = Path(__file__).resolve().parent
        config_path = (base_dir / relative_path).resolve()

        with open(config_path, "r") as f:
            self.data = yaml.safe_load(f)

    def __getitem__(self, key):
        return self.data[key]


def get_config_filenames(relative_path="conf/experiments/"):
    '''
    Get a list of configuration filenames in the specified directory.
    '''
    base_dir = Path(__file__).resolve().parent
    config_dir = (base_dir / relative_path).resolve()
    return [f.name for f in config_dir.glob("*.yaml")]


def get_config(relative_path="conf/settings.yaml"):
    '''
    Get the configuration instance, creating it if necessary.
    If no path provided, defaults to "conf/settings.yaml" relative to this file.
    '''
    config_instance = Config(relative_path)
    return config_instance
