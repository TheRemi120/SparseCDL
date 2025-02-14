import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator, lsmr, lsqr, cg
from numba import njit
from alphacsc import learn_d_z

# Global solver variable (default: lsmr)
solver_func = lsmr  
iteration_count = 0

def monCallback(x=None):
    """Callback function to increment the iteration count."""
    global iteration_count
    iteration_count += 1

@njit
def matvecMultiSparse(D, non_zero_array, Z, Td, T, output_dim, n_atoms):
    """
    Computes the matrix-vector product for a sparse operator.
    
    Parameters:
      D            : Flattened dictionary array.
      non_zero_array: Array of nonzero indices from Z.
      Z            : Activation tensor.
      Td           : Atom length.
      T            : Total length (signal length).
      output_dim   : Dimension of the output.
      n_atoms      : Number of atoms.
    
    Returns:
      A vector of shape (output_dim,) resulting from the operation.
    """
    result = np.zeros(output_dim)
    for idx in range(non_zero_array.shape[0]):
        n, i, k = non_zero_array[idx]
        atom_idx = i % n_atoms
        D_segment = D[atom_idx * Td:(atom_idx + 1) * Td]
        start = k + n * T
        end = start + Td
        result[start:end] += D_segment * Z[n, i, k]
    return result

@njit
def rmatvecMultiSparse(s, non_zero_array, Z, Td, T, input_dim, n_atoms):
    """
    Computes the transpose matrix-vector product for a sparse operator.
    
    Parameters:
      s            : Input vector.
      non_zero_array: Array of nonzero indices from Z.
      Z            : Activation tensor.
      Td           : Atom length.
      T            : Total length (signal length).
      input_dim    : Dimension of the input.
      n_atoms      : Number of atoms.
    
    Returns:
      A vector of shape (input_dim,) resulting from the operation.
    """
    result = np.zeros(input_dim)
    for idx in range(non_zero_array.shape[0]):
        n, i, k = non_zero_array[idx]
        atom_idx = i % n_atoms
        start = k + n * T
        end = start + Td
        result[atom_idx * Td:(atom_idx + 1) * Td] += s[start:end] * Z[n, i, k]
    return result

def my_update_d(X, z_hat, n_times_atom, lambd0, ds_init, verbose=10, solver_kwargs=None, sample_weights=None):
    """
    Custom update function for D using a specific solver.
    
    This function reshapes and processes the activations,
    builds a sparse linear operator, and solves for the updated dictionary.
    
    Parameters:
      X            : Input signal array.
      z_hat        : Activation tensor.
      n_times_atom : Expected time length of the activation.
      lambd0       : Regularization parameter.
      ds_init      : Initial dictionary.
      verbose      : Verbosity level.
      solver_kwargs: Additional solver arguments.
      sample_weights: Optional sample weights.
    
    Returns:
      Updated dictionary and the regularization parameter.
    """
    global solver_func, iteration_count
    print("Updating D")
    Z = np.transpose(z_hat, (1, 0, 2))

    n_signals = Z.shape[0]
    n_atoms = ds_init.shape[0]
    Tz = Z.shape[2]
    Td = ds_init.shape[1]
    T = Tz + Td - 1

    # Get indices where Z is nonzero
    i, j, k = np.where(Z)
    non_zero_array = np.column_stack((i, j, k))

    input_dim = Td * n_atoms
    output_dim = T
    multi_output_dim = T * n_signals

    # Define a linear operator for the sparse matrix multiplication
    A = LinearOperator(
        (multi_output_dim, input_dim),
        matvec=lambda x: matvecMultiSparse(x, non_zero_array, Z, Td, T, multi_output_dim, n_atoms),
        rmatvec=lambda x: rmatvecMultiSparse(x, non_zero_array, Z, Td, T, input_dim, n_atoms)
    )

    xFlatten = X.flatten()
    ATA = A.T @ A
    ATb = A.T @ xFlatten

    start_time = time.time()

    # Use the chosen solver to compute the updated dictionary
    if solver_func in [lsmr, lsqr]:
        result = solver_func(ATA, ATb)
        flatDictionary = result[0]
        iteration_count = result[2]
    else:
        flatDictionary = solver_func(ATA, ATb, callback=monCallback)[0]

    end_time = time.time()
    print(f"{solver_func.__name__} took {end_time - start_time:.3f} seconds")

    d_hat = flatDictionary.reshape(n_atoms, Td)
    norms = np.linalg.norm(d_hat, axis=1, keepdims=True)
    d_hat /= norms

    return d_hat, lambd0

def reconstructMySignal(d_hat, z_hat):
    """
    Reconstructs the original signal using the learned dictionary and activations.
    
    Parameters:
      d_hat : Learned dictionary (atoms).
      z_hat : Activation tensor.
    
    Returns:
      The reconstructed signal as an integer array.
    """
    n_atoms, n_trials, valid_length = z_hat.shape
    atom_length = d_hat.shape[1]
    total_length = valid_length + atom_length - 1

    reconstructed = np.zeros((n_trials, total_length))
    for k in range(n_atoms):
        for trial in range(n_trials):
            # Convolve each atom with its activation
            reconstructed[trial] += np.convolve(z_hat[k, trial], d_hat[k], mode='full')
    return np.int16(reconstructed)

def fit_D_Z(Y, n_atoms, n_times_atom, reg, n_iter, solver_z="ista", random_state=None, n_jobs=1, verbose=10):
    """
    Learns the dictionary (D) and activations (Z) using alphacsc.learn_d_z
    with our custom update function for D.
    
    Parameters:
      Y           : Input signal array.
      n_atoms     : Number of dictionary atoms.
      n_times_atom: Length of each atom.
      reg         : Regularization parameter.
      n_iter      : Number of iterations.
      solver_z    : Solver for Z (e.g., "ista").
      random_state, n_jobs, verbose: Additional parameters.
    
    Returns:
      A tuple (pobj, times, d_hat, z_hat, reg, iteration_count) with the learning results.
    """
    global solver_func, iteration_count
    iteration_count = 0
    pobj, times, d_hat, z_hat, reg = learn_d_z(
        Y,
        n_atoms=n_atoms,
        n_times_atom=n_times_atom,
        reg=reg,
        n_iter=n_iter,
        func_d=my_update_d,
        solver_z=solver_z,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose
    )
    return pobj, times, d_hat, z_hat, reg, iteration_count
