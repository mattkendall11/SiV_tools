import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from transitions import return_field_amp, convert_lab_frame, plot_magnitude_polar
def spin_orbit_hamiltonian(l):
    '''
    :param l: spin orbit coupling constants
    :return: spin orbit hamiltonian
    '''
    i = 1j
    return np.array([[0,0,-l*i/2,0],
                    [0,0,0,l*i/2],
                    [l*i/2,0,0,0],
                     [0, -i*l/2,0,0]]
                    )

def jahn_teller_hamiltonian(x, y):
    '''
    :param x: Jahn-Teller coupling strengths for the x component
    :param y: Jahn-Teller coupling strengths for the y component
    :return: jahn teller hamiltonian
    '''
    return np.array([[x, 0, y, 0],
                     [0, x, 0, y],
                     [y, 0, -x, 0],
                     [0, y, 0, -x]])

def zeeman_hamiltonian(f, L, Bz):
    '''
    :param f: quenching factor of orbital
    :param L: gyrometric ratio
    :param Bz: z component of magnetic field
    :return: zeeman hamiltonian
    '''
    iBz_fL = 1j * f * L * Bz
    return np.array([[0, 0, iBz_fL, 0],
                     [0, 0, 0, iBz_fL],
                     [-iBz_fL, 0, 0, 0],
                     [0, -iBz_fL, 0, 0]])

def spin_zeeman_hamiltonian(S, Bx, By, Bz):
    '''
    :param S: spin factor
    :param Bx: x component of B field
    :param By: y component of B field
    :param Bz: z component of B field
    :return: spin -zeeman hamiltonian
    '''
    return S * np.array([[Bz, Bx - 1j * By, 0, 0],
                          [Bx + 1j * By, -Bz, 0, 0],
                          [0, 0, Bz, Bx - 1j * By],
                          [0, 0, Bx + 1j * By, -Bz]])

def strain_hamiltonian(a, b, d):
    '''

    :param a: alpha strain
    :param b: beta strain
    :param d: gamma strain
    :return: strained hamiltonian
    '''
    return np.array([[a - d, 0, b, 0],
                     [0, a - d, 0, b],
                     [b, 0, -a - d, 0],
                     [0, b, 0, -a - d]])

def get_energy_spectra(l, x, y, f, B, strain = False):
    muB = 9.2740100783e-24  # Bohr magneton in J/T
    hbar = 1.054571817e-34  # Reduced Planck's constant in J*s
    GHz = 1e9  # 1 GHz in Hz

    # Gyromagnetic ratios
    L = muB / hbar  # Orbital gyromagnetic ratio
    S = 2 * muB / hbar
    a,b,d = 0,0,0
    if strain ==True:
        a,b,d = -238e9, 238e9, 0

    #firstly contruct full hamiltonian

    H = (spin_orbit_hamiltonian(l) + jahn_teller_hamiltonian(x,y) + zeeman_hamiltonian(f, L, B[2]) +
         spin_zeeman_hamiltonian(S, B[0], B[1], B[2])+ strain_hamiltonian(a,b,d))

    #find eigenvalues and eigenvectors

    eigenvalues, eigenvectors = eigh(H)

    return eigenvalues, eigenvectors
