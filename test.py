import h5py
import numpy as np


with h5py.File('data_100_11-19_16-09.h5', 'r') as f:
    # Access parameter arrays
    B_values = f['B_values'][:]
    theta_values = f['theta_values'][:]
    phi_values = f['phi_values'][:]

    # Access c magnitudes
    c_magnitudes = f['c_magnitudes'][:]

    # Access ground state energies and eigenvectors
    ground_energies = f['ground_state']['energy_levels'][:]
    ground_eigenvectors = f['ground_state']['eigenvectors'][:]

    # Access excited state energies and eigenvectors
    excited_energies = f['excited_state']['energy_levels'][:]
    excited_eigenvectors = f['excited_state']['eigenvectors'][:]


