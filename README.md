# Membership Inference Attacks: Implementation and Evaluation

## Table of Contents

- [Membership Inference Attacks: Implementation and Evaluation](#membership-inference-attacks-implementation-and-evaluation)
  - [Table of Contents](#table-of-contents)
  - [Abstract](#abstract)
  - [Introduction](#introduction)
  - [Methodology](#methodology)
    - [Membership Inference Attacks From First Principles](#membership-inference-attacks-from-first-principles)
    - [Low-Cost High-Power Membership Inference Attacks](#low-cost-high-power-membership-inference-attacks)
  - [Implementation](#implementation)
    - [Project Structure](#project-structure)
    - [Modules](#modules)
  - [Dataset and Model](#dataset-and-model)
    - [Dataset](#dataset)
    - [Model](#model)
  - [Attacks Implementation](#attacks-implementation)
    - [Membership Inference Attacks From First Principles](#membership-inference-attacks-from-first-principles-1)
    - [Low-Cost High-Power Membership Inference Attacks](#low-cost-high-power-membership-inference-attacks-1)
  - [Usage](#usage)
    - [Setup](#setup)
    - [Running the Attacks](#running-the-attacks)
    - [Generating and Sending Submissions](#generating-and-sending-submissions)
  - [References](#references)

---

## Abstract

This project presents the implementation and evaluation of two Membership Inference Attack (MIA) methods, as described in:
1. **"Membership Inference Attacks From First Principles"** (LiRA), and 
2. **"Low-Cost High-Power Membership Inference Attacks"** (RMIA).

We implement both *offline* and *online* variants of these attacks, assess their performance under various settings, and demonstrate how small modifications in code can transition from offline to online modes.

---

## Introduction

In this repository, we explore how an adversary can determine whether a specific sample was part of a machine learning model's training set by examining only black-box or gray-box access to the model. The repository provides a unified framework in which:

- **LiRA** is implemented by computing logit distributions (IN vs. OUT) and comparing them via Gaussian-based likelihood-ratio tests.
- **RMIA** is implemented by building pairwise likelihood-ratio tests with a population set, approximating \(\Pr(x)\) from reference models, and then deriving a robust membership score.


---

## Methodology

### Membership Inference Attacks From First Principles
- **Core Idea**: Train shadow models with explicit inclusion or exclusion of target points, and estimate the difference in model outputs (e.g., logit distributions).  
- **Online vs. Offline**: Offline omits separate IN shadow models, using only OUT distributions and approximate transformations.

### Low-Cost High-Power Membership Inference Attacks
- **Core Idea**: Evaluate pairwise likelihood ratios with a population set, computing \(\Pr(x)\) via reference models.  
- **Offline**: Only OUT models plus linear scaling; **Online**: splits half the reference models to include \(x\), half to exclude.

---


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

---

## Dataset and Model

### Dataset

The experiments utilize a specific dataset tailored for membership inference studies. The dataset is divided into:

- **`priv_out.pt`**: Represents the private dataset used for training the target model.
- **`pub.pt`**: Serves as a reference dataset for shadow models and other auxiliary tasks.

Both datasets are serialized using PyTorch's `torch.save` mechanism and are loaded as instances of the `MembershipDataset` class.

### Model

The target model employed in these attacks is based on the ResNet-18 architecture, modified to accommodate the specific number of output classes pertinent to the dataset. The model architecture is defined in `models/resnet18.py`, and pre-trained weights are stored in `weights/01_MIA_67.pt`.

---

## Attacks Implementation

### Membership Inference Attacks From First Principles

- **Offline LiRA**  
  - Found in `attacks/offline_lira.py`.  
  - Trains K OUT shadow models, collects logit distributions, and applies a Gaussian model.  

- **Online LiRA**  
  - Found in `attacks/online_lira.py`.  
  - Combines shadow sets in a single training pass, tracking which points are in which shadow model, thus effectively approximating separate IN/OUT sets for each sample.

### Low-Cost High-Power Membership Inference Attacks

- **Offline RMIA**  
  - Found in `attacks/rmia.py`.  
  - Uses a linear scaling factor \(a\) to approximate \(\Pr(x)\) from purely OUT shadow models.  
  - Dominance fraction (fraction of `z` such that `ratio_x >= gamma * ratio_z`) is computed via a sorted array or pairwise.

- **Online RMIA**  
  - Adapts logic from `online_lira.py` to the RMIA setting; trains a set of reference models, but half may include a given sample \(x\), half exclude it, for unbiased \(\Pr(x)\).

---

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

To securely store your API token and server URL for submissions, create a `.env` file in the project root and add the following:

```plaintext
SUBMISSION_API_TOKEN="your_token"
SUBMISSION_API_URL="server_url"
```

This ensures that sensitive credentials are not hardcoded in your scripts.


1. **Configure `config.yaml`**

   Ensure that all paths and parameters in `configs/config.yaml` are correctly set according to your environment and requirements.

2. **Execute the Main Script**

   Run the `main.py` script with the desired attack type.

   ```bash
   python main.py --config configs/config.yaml --attack online_lira
   ```

   **Available Attack Types:**

   - `random_guess`
   - `online_lira`
   - `offline_lira`
   - `online_rmia`
   - `offline_rmia`
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

---

## References

1. Carlini, N., Chien, S., Nasr, M., Song, S., Terzis, A., & Tramer, F. (2022). *Membership Inference Attacks From First Principles*. 2022 IEEE Symposium on Security and Privacy (SP), 1897–1914. [arXiv](https://arxiv.org/pdf/2112.03570)
2. Zarifzadeh, S., Liu, P., & Shokri, R. (2024). *Low-Cost High-Power Membership Inference Attacks*. Forty-first International Conference on Machine Learning (ICML). [arXiv](https://arxiv.org/pdf/2312.03262)

