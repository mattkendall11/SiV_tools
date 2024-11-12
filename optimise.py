import numpy as np
from scipy.optimize import minimize
from model import get_energy_spectra
from transitions import return_field_amp

# Constants
lg = 815e9  # Spin-orbit coupling in ground state (Hz)
lu = 2355e9  # Spin-orbit coupling in excited state (Hz)
x_g, y_g = 65e9, 0  # Jahn-Teller coupling (ground state) (Hz)
x_u, y_u = 855e9, 0  # Jahn-Teller coupling (excited state) (Hz)
fg, fu = 0.15, 0.15  # Quenching factors for ground and excited states

# Strain parameters
strain_params_strained = {'a_g': -238e9, 'b_g': 238e9, 'd_g': 0, 'a_u': -76e9, 'b_u': 76e9, 'd_u': 0}


# Cost function
def cost(B):
    # Calculate energy spectra and eigenvectors for the given B-field
    Eg, Vg = get_energy_spectra(lg, x_g, y_g, fg, B)
    Ee, Ve = get_energy_spectra(lu, x_u, y_u, fu, B)

    # Compute field amplitudes for A1 and A2 transitions
    A1x, A1y, A1z = return_field_amp(Ve[0], Vg[0])
    A2x, A2y, A2z = return_field_amp(Ve[0], Vg[1])

    # Calculate the dot product (inner product) to assess orthogonality
    dot_product = np.dot([A1x, A1y, A1z], [A2x, A2y, A2z])

    # The objective is to make the dot product zero, so return the absolute value
    return abs(dot_product)

B_initial = [1,1,1]
# Define bounds for B-field components
bounds = [(0, 9), (0, 9), (0, 9)]  # Each component of B is within 0 to 9 T

# Use L-BFGS-B method to apply bounds
result = minimize(cost, B_initial, method='L-BFGS-B', bounds=bounds)

# Output result
if result.success:
    optimal_B = result.x
    print("Optimal B-field:", optimal_B)
    print("Minimum dot product (ideally zero):", result.fun)
else:
    print("Optimization failed:", result.message)