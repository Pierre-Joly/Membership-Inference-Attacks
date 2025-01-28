import argparse
from logging import config
import torch
import pandas as pd
import numpy as np
import os

from attacks.ensemble_rmia import Ensemble_RMIA
from attacks.online_lira import OnlineLiRA
from attacks.random_guess import RandomGuessAttack
from attacks.offline_lira import OfflineLiRA
from attacks.rmia import RMIA
from datasets.dataset import MembershipDataset
from models.resnet18 import ResNet18Model
from utils.logger import logger
from utils.device_manager import get_device
from utils.config_loader import load_config


def load_model(model_path: str, num_classes: int, device: torch.device) -> torch.nn.Module:
    """
    Load the trained model.

    Args:
        model_path (str): Path to the trained model weights.
        num_classes (int): Number of output classes.
        device (torch.device): Device to load the model on.

    Returns:
        torch.nn.Module: Loaded model.
    """
    model = ResNet18Model(num_classes=num_classes)
    if not os.path.exists(model_path):
        logger.error(f"Model file {model_path} does not exist.")
        raise FileNotFoundError(f"Model file {model_path} not found.")
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    logger.info(f"Loaded model from {model_path}")
    return model


def load_dataset(data_path: str) -> MembershipDataset:
    """
    Load the membership dataset.

    Args:
        data_path (str): Path to the dataset file.

    Returns:
        MembershipDataset: Loaded dataset.
    """
    if not os.path.exists(data_path):
        logger.error(f"Data file {data_path} does not exist.")
        raise FileNotFoundError(f"Data file {data_path} not found.")
    
    data: MembershipDataset = torch.load(data_path, map_location='cpu', weights_only=False)

    logger.info(f"Loaded dataset from {data_path}, total samples: {len(data)}")
    return data


def execute_attack(attack_type: str, model: torch.nn.Module, data: MembershipDataset, attack_config: dict) -> torch.Tensor:
    """
    Execute the specified attack and obtain membership scores.

    Args:
        attack_type (str): Type of attack to execute.
        model (torch.nn.Module): The trained model.
        data (MembershipDataset): The dataset to attack.
        attack_config (dict): Configuration parameters for the attack.
        device (torch.device): Device to run the attack on.

    Returns:
        torch.Tensor: Membership scores.
    """
    if attack_type == 'random_guess':
        attack = RandomGuessAttack()
    elif attack_type == 'online_lira':
        attack = OnlineLiRA(**attack_config)
    elif attack_type == 'offline_lira':
        attack = OfflineLiRA(**attack_config)
    elif attack_type == 'rmia':
        attack = RMIA(**attack_config)
    elif attack_type == 'ensemble_rmia':
        attack = Ensemble_RMIA(**attack_config)
    else:
        logger.error(f"Unknown attack type: {attack_type}")
        raise ValueError(f"Unknown attack type: {attack_type}")

    logger.info(f"Executing {attack_type} attack")
    scores = attack.run_attack(model, data)
    logger.info(f"Attack '{attack_type}' completed")
    return scores


def generate_submission_csv(ids: list, scores: np.array, output_path: str):
    """
    Create a submission CSV file from IDs and scores.

    Args:
        ids (list): List of unique identifiers.
        scores (np.array): Membership scores.
        output_path (str): Path to save the submission CSV.
    """
    logger.info("Creating submission CSV")
    submission_df = pd.DataFrame({
        'ids': ids,
        'score': scores
    })
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    logger.info(f"Submission CSV saved to {output_path}")


def main(args):
    """
    Main function to generate submission CSV based on the specified attack.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    # Load configuration
    config = load_config(args.config)

    # Setup device
    device = get_device()
    logger.info(f"Using device: {device}")

    # Load model
    model_path = config['model']['trained_weights']
    num_classes = config['model']['num_classes']
    model = load_model(model_path, num_classes, device)

    # Load dataset
    data_path = config['data']['path']
    data = load_dataset(data_path)

    # Execute attack
    attack_type = args.attack
    if attack_type not in config['attacks']:
        logger.error(f"Attack type '{attack_type}' not found in configuration.")
        raise ValueError(f"Attack type '{attack_type}' not found in configuration.")
    attack_config = config['attacks'][attack_type]
    scores = execute_attack(attack_type, model, data, attack_config, device)

    # Generate submission CSV
    submission_path = config['submission']['submission_path']
    generate_submission_csv(data.ids, scores, submission_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Submission CSV for Membership Inference Attacks")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to the configuration file')
    parser.add_argument('--attack', type=str, required=True, choices=['random_guess', 'online_lira', 'offline_lira', 'rmia'], help='Type of attack to execute')
    args = parser.parse_args()
    main(args)
