# Fashion MNIST Denoising

This repository contains a deep learning pipeline for denoising Fashion MNIST images.

## Environment Setup

This project uses `uv` for fast and reliable dependency management. To install the dependencies and set up the virtual environment, simply run:
`uv sync`
to sync the environment using the provided uv.lock and pyproject.toml in the codebase

Note: If you haven't installed uv yet, you can install it following the official Astral documentation.

## Training
The main entry point for training the model is `train.py`. The project uses Hydra for configuration management.

To start the training loop with the default configuration, simply run:

`uv run train.py`
