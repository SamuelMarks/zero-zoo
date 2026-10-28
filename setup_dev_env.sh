#!/bin/bash
# Installs zero-zoo and all local zero-* / compiler repositories in editable mode for development.

echo "Setting up local editable environment..."
pip install -e ../ml-switcheroo-ir
pip install -e ../ml-switcheroo-compiler
pip install -e ../ml-ecosystem-snapshots
pip install -e ../cdd-python
pip install -e ../zero-jax
pip install -e ../zero-flax
pip install -e ../zero-pytorch
pip install -e ../zero-keras
pip install -e ../zero-tensorflow
pip install -e ../zero-mlx
pip install -e ../zero-pax
pip install -e ../zero-grain
pip install -e ../zero-optax
pip install -e ../zero-orbax
pip install -e ../zero-chex
pip install -e .[test]
echo "Done!"
