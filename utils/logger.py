import logging
import logging.config
import yaml
import os

def setup_logging(default_path='config/logging_config.yaml', default_level=logging.INFO):
    """
    Setup logging configuration from a YAML file.

    Args:
        default_path (str, optional): Path to the logging config file. Defaults to 'config/logging_config.yaml'.
        default_level (int, optional): Default logging level. Defaults to logging.INFO.
    """
    if os.path.exists(default_path):
        with open(default_path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
        logging.warning(f"Logging configuration file {default_path} not found. Using basic configuration.")

# Initialize logger
setup_logging()
logger = logging.getLogger('mia')
