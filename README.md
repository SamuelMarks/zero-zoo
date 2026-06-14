# zero-zoo: The Proving Grounds

[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/SamuelMarks/zero-zoo/actions/workflows/ci.yml/badge.svg)](https://github.com/SamuelMarks/zero-zoo/actions)
[![Test Coverage](https://img.shields.io/badge/test_coverage-100%25-brightgreen.svg)](#)
[![Doc Coverage](https://img.shields.io/badge/doc_coverage-100%25-brightgreen.svg)](#)

Welcome to **zero-zoo**, the central verification and testing hub for the Abstract ML Compiler ecosystem. 

As the name implies, this repository houses a **zoo of models** to rigorously verify our custom ML toolchain. We guarantee that our frontend API shells produce mathematically identical outcomes compared to their native counterparts by running them side-by-side.

## 🦁 The Model Zoo & Dialects

To guarantee **float-for-float mathematical equivalence**, we maintain standardized, canonical implementations of several key architectures (MLPs, CNNs, NanoGPT, Gemma, Qwen).

Crucially, each model in the zoo is implemented across all of these native **dialects**:
- **Flax NNX**
- **Flax Linen**
- **Pax**
- **JAX**
- **PyTorch**
- **TensorFlow**
- **Keras**

## 🧩 API-Compatible Implementations

The native dialect implementations are run alongside our new API-compatible ecosystem shells. If a model compiles and runs in a `zero-*` framework, it behaves *exactly* as it would natively. 

The models are tested against their respective `zero-*` counterparts:
- [`zero-flax`](https://github.com/SamuelMarks/zero-flax)
- [`zero-jax`](https://github.com/SamuelMarks/zero-jax)
- [`zero-keras`](https://github.com/SamuelMarks/zero-keras)
- [`zero-mlx`](https://github.com/SamuelMarks/zero-mlx)
- [`zero-pax`](https://github.com/SamuelMarks/zero-pax)
- [`zero-pytorch`](https://github.com/SamuelMarks/zero-pytorch)
- [`zero-tensorflow`](https://github.com/SamuelMarks/zero-tensorflow)

## ⚙️ The Compiler & Internal Backends

All of the `zero-*` frontends trace operations down to a single, unified compiler backend:
**[`ml-switcheroo-compiler`](https://github.com/SamuelMarks/ml-switcheroo-compiler)**

The `ml-switcheroo-compiler` does the heavy lifting (automatic differentiation, shape inference, optimizations). Once the intermediate representation (IR) is built, the compiler features multiple internal execution backends, allowing you to choose exactly how the math is evaluated:
- `numpy`
- `jax`
- `mlx`
- `cupy`
- `dusk`
- `torch`

## ✨ Key Verification Features

- **Matrix Runner (`matrix_runner.py`):** An equivalence checking harness that normalizes states and executions across differing framework paradigms.
- **Golden Seed Testing (`golden_seed.py`):** Deterministic initialization and training loops (e.g., training for 10 steps) to assert `.allclose()` parity over time. This prevents chaotic divergence during backpropagation.
- **Headless Browser CI (`headless_ci.py`):** End-to-end validation pipelines that execute the exported IR in headless browser environments.
- **Deterministic Data Pipelines (`data_gen.py`, `data_loader.py`):** Ensures all models receive the exact same mock data inputs and batches.

## 🚀 Usage & Testing

To run the verification matrix locally:

```bash
# Set up the development environment
./setup_dev_env.sh

# Run the full test suite (pytest)
pytest tests/

# Or run tox for matrix testing across different configurations
tox
```

For more detailed technical information, please refer to our [Architecture Documentation](ARCHITECTURE.md).

---

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.