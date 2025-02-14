# borelliCDL

borelliCDL is a Python library for convolutional dictionary learning applied to signal processing. It integrates `alphacsc` but lets you control the dictionary update step, allowing flexibility in choosing solvers. It also tracks the number of iterations during dictionary updates, so you can analyze solver performance.

## Features

- **Custom Dictionary Update:** Uses a specific function (`my_update_d`) to update the dictionary, compatible with solvers like `lsmr`, `lsqr`, `cg`, etc.
- **Signal Reconstruction:** Allows reconstructing signals from learned atoms and activations.
- **Direct Integration with alphacsc:** Uses `alphacsc.learn_d_z` while keeping full control over the dictionary update process.
- **Tracks Iterations:** Returns the number of iterations required for convergence.

## Installation

Clone the repo and install manually:

```bash
git clone https://github.com/yourusername/borelliCDL.git
cd borelliCDL
pip install .
```

Or install directly with pip:

```bash
pip install git+https://github.com/yourusername/borelliCDL.git
```

## Usage Example

Here’s a quick example to learn a dictionary and reconstruct signals:

```python
import numpy as np
from borelliCDL import fit_D_Z, reconstructMySignal

# Example signals (replace with real data)
Y = np.random.randn(3, 1000)  # 3 signals, length 1000

# Dictionary learning parameters
n_atoms = 5
n_times_atom = 350
reg = 0.045
n_iter = 3

# Learn dictionary and activations
pobj, times, d_hat, z_hat, reg, iteration_count = fit_D_Z(Y, n_atoms, n_times_atom, reg, n_iter)

# Reconstruct signals
reconstructed_signals = reconstructMySignal(d_hat, z_hat)

# Display results
print("Objective:", pobj)
print("Computation times:", times)
print("Iteration count:", iteration_count)
print("Reconstructed signals shape:", reconstructed_signals.shape)
```

## Requirements

- Python 3.6+
- numpy
- scipy
- matplotlib
- numba
- alphacsc
- pandas

## Contributing

If you want to improve the code or add new features, open an issue or submit a pull request.

## License

MIT License.
