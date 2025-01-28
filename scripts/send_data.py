import argparse
import requests
import os
import sys

from utils.logger import logger
from utils.config_loader import load_config


def send_submission(submission_path: str, server_url: str, token: str):
    if not os.path.exists(submission_path):
        logger.error(f"Submission file {submission_path} does not exist.")
        raise FileNotFoundError(f"Submission file {submission_path} not found.")

    logger.info(f"Sending submission from {submission_path} to {server_url}")

    with open(submission_path, 'rb') as f:
        files = {'file': (os.path.basename(submission_path), f, 'text/csv')}
        headers = {"token": token}
        try:
            response = requests.post(server_url, files=files, headers=headers)
            response.raise_for_status()
            response_data = response.json()
            logger.info(f"Submission successful. Server response: {response_data}")

            # Handle different response scenarios
            if 'score' in response_data:
                logger.info(f"Your current leaderboard score: {response_data['score']}")
            elif 'message' in response_data:
                logger.info(f"Server message: {response_data['message']}")
            else:
                logger.info("Submission received successfully.")
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error occurred: {http_err}")
            logger.error(f"Server response: {response.text}")
            raise
        except requests.exceptions.RequestException as req_err:
            logger.error(f"Request exception occurred: {req_err}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise


def main(args):
    # Load configuration
    config = load_config(args.config)

    # Retrieve submission path and server details
    submission_path = config['submission']['submission_path']

    # Retrieve token securely from environment variable
    token = os.getenv('SUBMISSION_API_TOKEN')
    if not token:
        logger.error("API token not found. Please set the SUBMISSION_API_TOKEN environment variable.")
        sys.exit(1)

    # Retrieve serveur URL from environment variable
    server_url = os.getenv('SERVEUR_URL')
    if not server_url:  
        logger.error("URL not found. Please set the SERVEUR_URL environment variable.")
        sys.exit(1)

    # Send submission
    send_submission(submission_path, server_url, token)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send Submission CSV to Server")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to the configuration file')
    args = parser.parse_args()
    main(args)
