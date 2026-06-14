#!/bin/bash
# Installs zero-zoo and all local zero-* / compiler repositories in editable mode for development.

echo "Setting up local editable environment..."
pip install -e .[dev]
pip install -e ../ml-switcheroo-compiler
pip install -e ../zero-pytorch
pip install -e ../zero-jax
pip install -e ../zero-flax
pip install -e ../zero-keras
pip install -e ../zero-mlx
pip install -e ../zero-tensorflow
echo "Done!"
