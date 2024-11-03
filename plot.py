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
strain_params_strained = {'α_g': -238e9, 'β_g': 238e9, 'δ_g': 0, 'α_u': -76e9, 'β_u': 76e9, 'δ_u': 0}

# Magnetic constants

#B-field
B = [0,0,0]
B_values = np.arange(0,9.1,0.1)
Ee_values = np.zeros((4, len(B_values)))
Eg_values = np.zeros((4, len(B_values)))
for i in range(len(B_values)):
    B = [1,1,B_values[i]]
    Ee, Ve = get_energy_spectra(lu, x_u, y_u, fu, B)
    Eg, Vg = get_energy_spectra(lg,x_g,y_g,fg,B)
    Ee_values[:,i] = Ee
    Eg_values[:,i] = Eg

f ,(ax1,ax2) = plt.subplots(2,1, sharex=True)
for i in range(4):
    ax1.plot(B_values, Ee_values[i,:], color = 'r')

    ax2.plot(B_values, Eg_values[i,:], color = 'b')
ax1.set_title('excited state')
ax2.set_title('ground state')
plt.xlabel('Bz (T)')
plt.suptitle('E-level response to Bz, Bx=By=1T')
plt.show()

