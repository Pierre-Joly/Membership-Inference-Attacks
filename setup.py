from setuptools import setup, find_packages

setup(
    name='Membership_Inference_Attacks',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
        'tqdm',
        'pyyaml',
        'pandas',
        'requests'
    ],
)
