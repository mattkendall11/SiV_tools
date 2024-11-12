import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting tools
from model import get_energy_spectra
from transitions import return_field_amp

# Constants
lg = 815e9  # Spin-orbit coupling in ground state (Hz)
lu = 2355e9  # Spin-orbit coupling in excited state (Hz)
x_g, y_g = 65e9, 0  # Jahn-Teller coupling (ground state) (Hz)
x_u, y_u = 855e9, 0  # Jahn-Teller coupling (excited state) (Hz)
fg, fu = 0.15, 0.15  # Quenching factors for ground and excited states

# Define range for Bx and By
Bx_values = np.linspace(0, 9, 100)  # 50 points from 0 to 9 T for Bx
By_values = np.linspace(0, 9, 100)  # 50 points from 0 to 9 T for By

# Create meshgrid for Bx and By
Bx_grid, By_grid = np.meshgrid(Bx_values, By_values)
c_magnitudes = np.zeros_like(Bx_grid)

# Calculate |c| for each combination of Bx and By
for i in range(len(Bx_values)):
    for j in range(len(By_values)):
        B = [Bx_grid[i, j], By_grid[i, j], 0]  # Bz is fixed at 0

        # Get energy spectra and eigenvectors
        Eg, Vg = get_energy_spectra(lg, x_g, y_g, fg, B)
        Ee, Ve = get_energy_spectra(lu, x_u, y_u, fu, B)

        # Compute field amplitudes for A1 and A2 transitions
        A1x, A1y, A1z = return_field_amp(Ve[0], Vg[0])
        A2x, A2y, A2z = return_field_amp(Ve[0], Vg[1])

        # Calculate the complex dot product
        c = np.dot([A1x, A1y, A1z], [A2x, A2y, A2z])

        # Store the magnitude of c
        c_magnitudes[i, j] = np.abs(c)

# Plot the results as a 3D surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(Bx_grid, By_grid, c_magnitudes, cmap='viridis')

# Labels and title
ax.set_xlabel('Magnetic Field B_x (T)')
ax.set_ylabel('Magnetic Field B_y (T)')
ax.set_zlabel('|c| (Magnitude of Dot Product)')
ax.set_title(fr'$A_1 \dot A_2$')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

plt.show()
