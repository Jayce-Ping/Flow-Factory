#!/usr/bin/env python
# setup.py
"""
Flow-Factory: Unified RL Fine-tuning Framework for Diffusion Models
"""
from setuptools import setup, find_packages

# Read requirements
def parse_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="flow-factory",
    version="0.1.0",
    description="Unified RL Fine-tuning Framework for Diffusion/Flow-Matching Models",
    author="Flow-Factory Team",
    author_email="",
    url="https://github.com/your-org/flow-factory",
    
    # Package discovery
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Dependencies
    install_requires=[
        "torch==2.8.0",
        "torchvision==0.23.0",
        "torchaudio",
        "transformers==4.57.1",
        "accelerate==1.11.0",
        "diffusers==0.35.2",

        "deepspeed==0.17.4",
        "peft==0.17.1",
        "bitsandbytes==0.45.3",        
        "huggingface-hub==0.35.3",
        "tokenizers==0.22.1",

        "datasets==3.3.2",
        "numpy==1.26.4",
        "pandas==2.2.3",
        "scipy==1.15.2",
        "scikit-learn==1.6.1",
        "scikit-image==0.25.2",
        "open-clip-torch==3.1.0",
        
        "albumentations==1.4.10",  
        "opencv-python==4.11.0.86",
        "pillow==10.4.0",
        
        "tqdm",
        "pydantic==2.10.6",  
        "requests==2.32.3",
        "matplotlib==3.10.0",
        "aiohttp==3.11.13",
        "fastapi==0.115.11", 
        "uvicorn==0.34.0",
        "einops==0.8.1",
        "nvidia-ml-py==12.570.86",
        "xformers",
        "absl-py",
        "sentencepiece",
    ],
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
        "reward": [
            "ImageReward>=1.0.0",
        ],
    },
    
    # Entry points - command line scripts
    entry_points={
        "console_scripts": [
            "flow-factory-train=flow_factory.cli:train_cli",
            "flow-factory-eval=flow_factory.cli:eval_cli",
        ],
    },
    
    # Package metadata
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    
    # Include package data
    include_package_data=True,
    zip_safe=False,
)