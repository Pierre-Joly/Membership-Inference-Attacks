import argparse
import os
import sys

from utils.key_manager import load_env_from_file
from utils.logger import logger
from utils.config_loader import load_config
from utils.device_manager import get_device
from scripts.generate_data import load_dataset, load_model, execute_attack, generate_submission_csv
from scripts.send_data import send_submission
from datasets.dataset import MembershipDataset


def main():
    """
    Main function to generate submission CSV and send it to the server.
    """
    parser = argparse.ArgumentParser(description="Generate and Send Submission for Membership Inference Attacks")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to the configuration file')
    parser.add_argument('--attack', type=str, required=True, choices=['random_guess', 'online_lira', 'offline_lira', 'offline_rmia'], help='Type of attack to execute')
    args = parser.parse_args()

    try:
        # Load configuration
        config = load_config(args.config)

        # Setup device
        device = get_device()
        logger.info(f"Using device: {device}")

        # Load model
        model = load_model(config['model']['trained_weights'], config['model']['num_classes'], device)

        # Load dataset
        data = load_dataset(config['data']['path'])

        # Execute attack
        scores = execute_attack(args.attack, model, data, config['attacks'][args.attack])

        # Generate submission CSV
        generate_submission_csv(data.ids, scores, config['submission']['submission_path'])

        # Load environment variables
        load_env_from_file()

        # Retrieve API token from environment variable
        api_token = os.getenv('SUBMISSION_API_TOKEN')
        if not api_token:
            logger.error("API token not found. Please set the SUBMISSION_API_TOKEN environment variable.")
            sys.exit(1)

        # Retrieve serveur URL from environment variable
        server_url = os.getenv('SUBMISSION_API_URL')
        if not server_url:  
            logger.error("URL not found. Please set the SUBMISSION_API_URL environment variable.")
            sys.exit(1)

        # Send submission
        send_submission(config['submission']['submission_path'], server_url, api_token)

    except Exception as e:
        logger.error(f"An error occurred during the submission process: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
