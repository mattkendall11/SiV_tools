import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from model import get_energy_spectra
from transitions import get_magnitude_from_vector, plot_magnitude_polar

# Constants
lg = 815e9  # Spin-orbit coupling in ground state (Hz)
lu = 2355e9  # Spin-orbit coupling in excited state (Hz)
x_g, y_g = 65e9, 0  # Jahn-Teller coupling (ground state) (Hz)
x_u, y_u = 855e9, 0  # Jahn-Teller coupling (excited state) (Hz)
fg, fu = 0.15, 0.15  # Quenching factors for ground and excited states

# Strain parameters
strain_params_strained = {'a_g': -238e9, 'b_g': 238e9, 'd_g': 0, 'a_u': -76e9, 'b_u': 76e9, 'd_u': 0}

# Initial B-field vector
B_initial = [1, 0.5, 9]

# Target angle for major axis rotation
target_angle = 90.0  # degrees
indx, indx1 = 2, 3

# Optimization function to rotate the major axis
def objective(B):
    # Calculate energy spectra and vectors for given B
    Ee, Ve = get_energy_spectra(lu, x_u, y_u, fu, B)
    Eg, Vg = get_energy_spectra(lg, x_g, y_g, fg, B)

    # Calculate phi and magnitudes from vectors
    phi_values, magnitudes = get_magnitude_from_vector(Ve[indx], Vg[indx1])

    # Convert to Cartesian
    Ex = magnitudes * np.cos(phi_values)
    Ey = magnitudes * np.sin(phi_values)

    # Find covariance matrix and eigenvalues/eigenvectors for axes
    cov_matrix = np.cov(Ex, Ey)
    eigvals, eigvecs = np.linalg.eig(cov_matrix)

    # Calculate major axis angle
    major_axis_direction = eigvecs[:, 0]
    major_axis_angle = np.degrees(np.arctan2(major_axis_direction[1], major_axis_direction[0]))

    # Return the difference from the target angle
    return (major_axis_angle - target_angle) ** 2


# Run optimization
result = minimize(objective, B_initial, method="Nelder-Mead")
B_optimized = result.x
print(f"Optimized B-field vector: {B_optimized}")

# Calculate the final rotation using the optimized B
Ee, Ve = get_energy_spectra(lu, x_u, y_u, fu, B_optimized)
Eg, Vg = get_energy_spectra(lg, x_g, y_g, fg, B_optimized)
phi_values, magnitudes = get_magnitude_from_vector(Ve[indx], Vg[indx1])
Ex = magnitudes * np.cos(phi_values)
Ey = magnitudes * np.sin(phi_values)

# Plot the polarization ellipse with optimized B
fig, ax = plt.subplots()
ax.plot(Ex, Ey, label="Polarization Ellipse (Optimized B)")

# Find and plot the new major and minor axes
cov_matrix = np.cov(Ex, Ey)
eigvals, eigvecs = np.linalg.eig(cov_matrix)
major_axis_direction = eigvecs[:, 0]
major_axis_length = 2 * np.sqrt(eigvals[0])
minor_axis_length = 2 * np.sqrt(eigvals[1])

# Plot major and minor axes
origin = [0, 0]
ax.quiver(*origin, *major_axis_direction * major_axis_length, color="r", scale=1, scale_units="xy",
          label="Major Axis (Optimized)")
ax.quiver(*origin, *eigvecs[:, 1] * minor_axis_length, color="g", scale=1, scale_units="xy", label="Minor Axis")

# Final major axis angle
final_angle = np.degrees(np.arctan2(major_axis_direction[1], major_axis_direction[0]))
print(f"Final major axis angle: {final_angle:.2f}°")

# Format and display the plot
ax.legend()
ax.set_aspect('equal')
plt.title("Polarization Plot with Optimized Major Axis")
plt.grid(True)
plt.show()
