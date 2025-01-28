import os

def load_env_from_file(env_file: str = ".env"):
    """
    Load environment variables from a .env file into the system environment.

    Args:
        env_file (str): Path to the .env file. Defaults to '.env'.

    Raises:
        FileNotFoundError: If the .env file does not exist.
    """
    if not os.path.exists(env_file):
        raise FileNotFoundError(f"The environment file '{env_file}' was not found.")
    
    with open(env_file, "r") as file:
        for line in file:
            # Skip empty lines and comments
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            # Split key-value pairs
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")  # Remove quotes if present
            
            # Set the environment variable
            os.environ[key] = value
            print(f"Loaded environment variable: {key}")