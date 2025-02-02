import logging.config
import os
import yaml

def setup_logging(default_path='config/logging_config.yaml', default_level=logging.INFO):
    """
    Set up logging configuration from a YAML file, ensuring only one log file is created.

    Args:
        default_path (str, optional): Path to the logging config file. Defaults to 'config/logging_config.yaml'.
        default_level (int, optional): Default logging level. Defaults to logging.INFO.
    """

    # Ensure the logs directory exists
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)


    if os.path.exists(default_path):
        with open(default_path, 'rt') as f:
            config = yaml.safe_load(f.read())

        logging.config.dictConfig(config)  # Apply the updated configuration
    else:
        # Use basic configuration if the logging config file is missing
        logging.basicConfig(level=default_level)
        logging.warning(f"Logging configuration file {default_path} not found. Using basic configuration.")

# Ensure the logger setup happens once
setup_logging()

# Get the logger instance
logger = logging.getLogger()
