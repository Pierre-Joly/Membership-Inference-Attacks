import yaml
import os
from utils.logger import logger

def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        logger.error(f"Configuration file {config_path} does not exist.")
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {config_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading configuration: {e}")
        raise
