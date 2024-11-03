import numpy as np
import matplotlib.pyplot as plt
from model import get_energy_spectra
from transitions import get_magnitude_from_vector, plot_magnitude_polar
'''
define params
'''
# Constants
lg = 815e9        # Spin-orbit coupling in ground state (Hz)
lu = 2355e9       # Spin-orbit coupling in excited state (Hz)
x_g, y_g = 65e9, 0  # Jahn-Teller coupling (ground state) (Hz)
x_u, y_u = 855e9, 0  # Jahn-Teller coupling (excited state) (Hz)
fg, fu = 0.15, 0.15  # Quenching factors for ground and excited states

# Strain parameters
strain_params_strained = {'a_g': -238e9, 'b_g': 238e9, 'd_g': 0, 'a_u': -76e9, 'b_u': 76e9, 'd_u': 0}

# Magnetic constants

#B-field
fig, ax = plt.subplots(4, 4, figsize=(15, 15), subplot_kw={'projection': 'polar'})
labels = ['a', 'b', 'c', 'd']
colors = ["blue", "green", "red", "cyan", "magenta", "yellow", "black", "white", "orange", "purple"]
Bz_values = np.arange(0,9,1)
for b in Bz_values:
    B = [b,0,0]


    Ee, Ve = get_energy_spectra(lu, x_u, y_u, fu, B)
    Eg, Vg = get_energy_spectra(lg,x_g,y_g,fg,B)

    # Create plots for each combination of Ve and Vg
    for i in range(4):
        for j in range(4):
            phi_values, magnitudes = get_magnitude_from_vector(Ve[i], Vg[j])

            # Plot magnitude vs phi
            ax[i, j].plot(phi_values, magnitudes, color = colors[b], label = fr'B_x = {b}T')
            if b == 0.2:
            # Customize the plot
                ax[i, j].grid(True)
                ax[i, j].set_xticklabels([])
                ax[i, j].set_yticklabels([])
                # Add labels

                ax[i, j].set_title(labels[i]+str(j+1), fontsize=8)

plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1))  # Move it outside the plot area

plt.suptitle('polarisations of all 16 transitions at 0T')
plt.tight_layout()

plt.show()
# plot_magnitude_polar(phi_values, magnitudes)
