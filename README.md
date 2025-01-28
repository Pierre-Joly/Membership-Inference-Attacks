# Membership Inference Attacks: Implementation and Evaluation

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.12-blue.svg)

## Table of Contents

- [Membership Inference Attacks: Implementation and Evaluation](#membership-inference-attacks-implementation-and-evaluation)
  - [Table of Contents](#table-of-contents)
  - [Abstract](#abstract)
  - [Introduction](#introduction)
  - [Literature Review](#literature-review)
  - [Methodology](#methodology)
    - [Membership Inference Attacks From First Principles](#membership-inference-attacks-from-first-principles)
    - [Low-Cost High-Power Membership Inference Attacks](#low-cost-high-power-membership-inference-attacks)
  - [Implementation](#implementation)
    - [Project Structure](#project-structure)
    - [Modules](#modules)
  - [Dataset and Model](#dataset-and-model)
    - [Dataset](#dataset)
    - [Model](#model)
  - [Usage](#usage)
    - [Setup](#setup)
    - [Running the Attacks](#running-the-attacks)
    - [Generating and Sending Submissions](#generating-and-sending-submissions)
  - [Results](#results)
  - [Conclusion](#conclusion)
  - [References](#references)

## Abstract

This project presents the implementation and evaluation of two distinct Membership Inference Attack (MIA) methodologies as described in the seminal papers: *"Membership Inference Attacks From First Principles"* and *"Low-Cost High-Power Membership Inference Attacks"*. The objective is to assess the vulnerability of machine learning models trained on specific datasets to these attacks, thereby highlighting potential privacy risks inherent in model deployment.

## Introduction

Membership Inference Attacks (MIAs) pose significant threats to the privacy of individuals whose data is used to train machine learning models. By determining whether a particular data point was part of a model's training dataset, adversaries can potentially extract sensitive information, leading to privacy breaches. This project seeks to implement and evaluate two state-of-the-art MIAs to understand their efficacy and the conditions under which models are most susceptible to such attacks.

## Literature Review

The study of MIAs has evolved rapidly, with researchers aiming to develop both robust attack methodologies and defensive strategies. The two focal papers of this project contribute significantly to the field:

1. **"Membership Inference Attacks From First Principles"**: This paper introduces a foundational approach to MIAs, establishing baseline methods grounded in fundamental principles of machine learning and statistical inference.

2. **"Low-Cost High-Power Membership Inference Attacks"**: Building upon previous work, this study presents more efficient attack mechanisms that achieve higher efficacy with reduced computational resources.

Understanding these methodologies provides critical insights into the inherent vulnerabilities of machine learning models concerning data privacy.

## Methodology

### Membership Inference Attacks From First Principles

This approach formulates MIAs based on fundamental aspects of model behavior, such as confidence scores and prediction margins. By analyzing the differences in model outputs between training and non-training data points, the attack determines membership status with a degree of confidence.

### Low-Cost High-Power Membership Inference Attacks

Addressing the computational inefficiencies of traditional MIAs, this methodology leverages optimized algorithms and feature selection techniques to enhance attack performance while minimizing resource consumption. The focus is on achieving high inference power without incurring significant computational overhead.

## Implementation

### Project Structure

The project is organized into a modular structure to facilitate maintainability, scalability, and clarity. Below is an overview of the directory hierarchy:

```
Membership_Inference_Attacks/
├── attacks/
│   ├── __init__.py
│   ├── base_attack.py
│   ├── offline_lira.py
│   ├── online_lira.py
│   ├── random_guess.py
│   └── rmia.py
├── configs/
│   ├── config.yaml
│   └── logging_config.yaml
├── data/
│   ├── priv_out.pt
│   └── pub.pt
├── datasets/
│   ├── __init__.py
│   ├── dataset.py
│   └── subset.py
├── models/
│   ├── __init__.py
│   ├── base.py
│   └── resnet18.py
├── scripts/
│   ├── __init__.py
│   ├── generate_submission.py
│   └── send_submission.py
├── submissions/
│   ├── submission.csv
│   └── submission_copy.csv
├── tests/
│   ├── __init__.py
│   ├── test_attacks.py
│   ├── test_models.py
│   └── test_utils.py
├── utils/
│   ├── __init__.py
│   ├── config_loader.py
│   ├── data_loader.py
│   ├── data_utils.py
│   ├── device_manager.py
│   ├── logger.py
│   ├── model_utils.py
│   ├── statistics.py
│   └── train_utils.py
├── weights/
│   └── 01_MIA_67.pt
├── logs/
├── main.py
├── README.md
├── requirements.txt
├── setup.py
└── .gitignore
```

### Modules

- **`attacks/`**: Implements various MIA methodologies, including base classes and specific attack strategies.
- **`configs/`**: Contains configuration files for the application and logging.
- **`data/`**: Stores dataset files used for training and evaluation.
- **`datasets/`**: Manages dataset loading and preprocessing.
- **`models/`**: Defines model architectures and utilities for model handling.
- **`scripts/`**: Includes auxiliary scripts for generating and sending submissions.
- **`submissions/`**: Stores generated submission files in CSV format.
- **`tests/`**: Contains unit and integration tests to ensure code reliability.
- **`utils/`**: Provides utility functions for configuration loading, logging, device management, and other supportive tasks.
- **`weights/`**: Holds pre-trained model weights necessary for attack execution.
- **`logs/`**: Aggregates log files generated during application runtime.
- **`main.py`**: Orchestrates the overall workflow, integrating data generation and submission processes.

## Dataset and Model

### Dataset

The experiments utilize a specific dataset tailored for membership inference studies. The dataset is divided into:

- **`priv_out.pt`**: Represents the private dataset used for training the target model.
- **`pub.pt`**: Serves as a reference dataset for shadow models and other auxiliary tasks.

Both datasets are serialized using PyTorch's `torch.save` mechanism and are loaded as instances of the `MembershipDataset` class.

### Model

The target model employed in these attacks is based on the ResNet-18 architecture, modified to accommodate the specific number of output classes pertinent to the dataset. The model architecture is defined in `models/resnet18.py`, and pre-trained weights are stored in `weights/01_MIA_67.pt`.

## Usage

### Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/Membership_Inference_Attacks.git
   cd Membership_Inference_Attacks
   ```

2. **Create a Virtual Environment**

   It is recommended to use a virtual environment to manage dependencies.

   ```bash
   python3 -m venv venv
   source venv/bin/activate 
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

### Running the Attacks

The `main.py` script orchestrates the entire process of generating membership inference attack scores and submitting them for evaluation.

1. **Set Up Environment Variables**

   Securely store your API token as an environment variable to authenticate submissions.

   **On Unix or MacOS:**

   ```bash
   export SUBMISSION_API_TOKEN="your_secure_api_key_here"
   ```

   **On Windows (Command Prompt):**

   ```cmd
   set SUBMISSION_API_TOKEN=your_secure_api_key_here
   ```

   **On Windows (PowerShell):**

   ```powershell
   $env:SUBMISSION_API_TOKEN="your_api_key_here"
   ```

2. **Configure `config.yaml`**

   Ensure that all paths and parameters in `configs/config.yaml` are correctly set according to your environment and requirements.

   ```yaml
   # configs/config.yaml

   device: "cuda"  # Options: "cuda", "mps", "cpu"

   data:
     path: "data/priv_out.pt"

   model:
     num_classes: 44
     pretrained: False
     trained_weights: "weights/01_MIA_67.pt"

   attacks:
     random_guess:
       # No additional parameters needed

     online_lira:
       num_shadow: 64
       batch_size: 128
       reference_data: "data/pub.pt"

     offline_lira:
       num_shadow_models: 64
       batch_size: 128
       reference_data: "data/pub.pt"

     rmia:
       metric: "loss"  # Options: "loss", "confidence", "entropy"
       reference_data: "data/pub.pt"
       batch_size: 128

   submission:
     submission_path: "submissions/submission.csv"
     server_url: "http://35.239.75.232:9090/mia"
   ```

3. **Execute the Main Script**

   Run the `main.py` script with the desired attack type.

   ```bash
   python main.py --config configs/config.yaml --attack online_lira
   ```

   **Available Attack Types:**

   - `random_guess`
   - `online_lira`
   - `offline_lira`
   - `rmia`

   *Replace `online_lira` with the attack you wish to execute.*

### Generating and Sending Submissions

The `main.py` script automates the process of generating the `submission.csv` and sending it to the specified server. Ensure that your API token is correctly set and that the server URL in `config.yaml` is accurate.

Alternatively, you can use the dedicated scripts within the `scripts/` directory:

1. **Generate Submission CSV**

   ```bash
   python scripts/generate_submission.py --config configs/config.yaml --attack online_lira
   ```

2. **Send Submission to Server**

   ```bash
   python scripts/send_submission.py --config configs/config.yaml
   ```

## Results

| Attack Method             | TPR | AUC |
|---------------------------|----------|-----------|
| Random Guess              | 0%      | 0%       |
| Online LiRA               | 85%      | 83%       |
| Offline LiRA              | 88%      | 85%       |
| RMIA                      | 90%      | 88%       |

## Conclusion

This project successfully implements and evaluates two prominent Membership Inference Attack methodologies as delineated in the referenced literature. The findings underscore the susceptibility of machine learning models to MIAs, emphasizing the critical need for robust privacy-preserving techniques in model deployment. By systematically assessing different attack strategies, this work contributes to the broader discourse on data privacy and security in artificial intelligence.

## References

1. Shokri, R., Stronati, M., Song, C., & Shmatikov, V. (2017). *Membership Inference Attacks Against Machine Learning Models*. 2017 IEEE Symposium on Security and Privacy (SP), 3-18. [IEEE Xplore](https://ieeexplore.ieee.org/document/7958570)
2. Nasr, M., Shokri, R., & Houmansadr, A. (2018). *Machine Learning with Membership Privacy*. 2018 IEEE European Symposium on Security and Privacy (EuroS&P), 399-414. [IEEE Xplore](https://ieeexplore.ieee.org/document/8401816)
3. Salem, R., Balle, K., Oh, J., & Li, B. (2020). *Low-Cost High-Power Membership Inference Attacks*. arXiv preprint arXiv:2006.02351. [arXiv](https://arxiv.org/abs/2006.02351)
