import numpy as np
from scipy.linalg import eigh, kron


def calculate_level_splitting(G1, G2):
    """Calculate level splitting in the excited (EEe) and ground state (EEg)"""
    GHz = 1e9
    EEe = (G1 + G2) / 4
    EEg = (G1 - G2) / 4
    return EEe, EEg


def calculate_SO_JT_contributions(EE, theta, phi):
    """Calculate SO (D) and JT (Q) interaction contributions"""
    D = 2 * EE * np.cos(theta)
    Qx = EE * np.sin(theta) * np.cos(phi)
    Qy = EE * np.sin(theta) * np.sin(phi)
    Q = np.sqrt(Qx ** 2 + Qy ** 2)
    return D, Q, Qx, Qy


def calculate_energy_splitting(Q, D):
    """Calculate the total energy splitting (DeltaE)"""
    GHz = 1e9
    DeltaE = 2 * np.sqrt(Q ** 2 + (D ** 2) / 4) / GHz
    return DeltaE


def create_hamiltonian(D, Bv, Qx, Qy, f, eps):
    """Construct the Hamiltonian including all interactions (SO, JT, Zeeman, stress)"""
    ge = 28e9
    gL = ge / 2

    Lx = np.array([[0, 1], [1, 0]])
    Ly = np.array([[0, -1j], [1j, 0]])
    Lz = np.diag([1, -1])

    Sx = np.array([[0, 1], [1, 0]]) / 2
    Sy = np.array([[0, -1j], [1j, 0]]) / 2
    Sz = np.array([[1, 0], [0, -1]]) / 2
    I2 = np.identity(2)

    Bx, By, Bz = Bv
    H = (-D * kron(Ly, Sz) +
         f * gL * kron(Bz * Ly, I2) +
         ge * kron(I2, Bz * Sz) +
         ge * kron(I2, Bx * Sx) +
         ge * kron(I2, By * Sy) +
         Qx * kron(Lz, I2) +
         Qy * kron(Lx, I2))

    H_stress = create_stress_hamiltonian(eps)
    return H + H_stress


def create_stress_hamiltonian(eps):
    """Generate the uniaxial stress Hamiltonian along the [100] direction"""
    I2 = np.identity(2)
    Hstress2dim = eps * np.array([[-1, np.sqrt(3)], [np.sqrt(3), 1]])
    HStressU = kron(Hstress2dim, I2)
    return HStressU


def calculate_eigenvalues_eigenvectors(H):
    """Compute eigenvalues and eigenvectors of the Hamiltonian H"""
    eigenvalues, eigenvectors = eigh(H)
    return eigenvalues, eigenvectors


def simulate_model(G1, G2, tg, pg, te, pe, tB, f, eps):
    """
    Main function to simulate the model using specified parameters and returns calculated values.

    Parameters:
    - G1, G2: Spectral splittings in Hz
    - tg, pg, te, pe: Angles for ground and excited states
    - tB: Angle between high symmetry axis and magnetic field
    - f: Factor for orbital g-factor reduction
    - eps: Uniaxial stress

    Returns:
    - Dictionary of calculated energy levels, transitions, and spectra.
    """
    GHz = 1e9
    B_field = np.arange(0, 7.1, 0.1)  # Magnetic field vector
    t = np.radians(tB)
    p = np.radians(45)

    G1, G2 = G1 * GHz, G2 * GHz
    eps = eps * 242 * 2 * GHz  # Example scaling

    EEe, EEg = calculate_level_splitting(G1, G2)
    Dg, Qg, Qgx, Qgy = calculate_SO_JT_contributions(EEg, tg, pg)
    DeltaEg = calculate_energy_splitting(Qg, Dg)
    De, Qe, Qex, Qey = calculate_SO_JT_contributions(EEe, te, pe)
    DeltaEe = calculate_energy_splitting(Qe, De)

    results = {
        'DeltaEg': DeltaEg,
        'DeltaEe': DeltaEe,
        'B_field': B_field,
        'Eg': [],
        'Ee': [],
        'T': [],
        'I': []
    }

    for b in B_field:
        Bv = b * np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])
        Hg = create_hamiltonian(Dg, Bv, Qgx, Qgy, f, eps)
        He = create_hamiltonian(De, Bv, Qex, Qey, f, 1.3 * eps)

        Eg_vals, Vg = calculate_eigenvalues_eigenvectors(Hg)
        Ee_vals, Ve = calculate_eigenvalues_eigenvectors(He)
        results['Eg'].append(Eg_vals)
        results['Ee'].append(Ee_vals)

        Ta, Ia = calculate_transitions(Eg_vals, Ee_vals, Vg, Ve)
        results['T'].append(Ta)
        results['I'].append(Ia)

    return results


def calculate_transitions(Eg_vals, Ee_vals, Vg, Ve):
    """Calculate transition frequencies and intensities based on eigenvalues"""
    etaX, etaY, etaZ = 1, 0.642, 0.823
    kB = 1.38065e-23 / 6.6261e-34  # Boltzmann constant in units of Hz/K

    T = []
    I = []
    for g, Eg in enumerate(Eg_vals):
        for e, Ee in enumerate(Ee_vals):
            transition_energy = Ee - Eg
            T.append(transition_energy)

            vg = Vg[:, g] / np.linalg.norm(Vg[:, g])
            ve = Ve[:, e] / np.linalg.norm(Ve[:, e])

            Ax = abs(vg.T @ np.kron(np.array([[1, 0], [0, -1]]), np.identity(2)) @ ve) ** 2
            Ay = abs(vg.T @ np.kron(np.array([[0, -1], [-1, 0]]), np.identity(2)) @ ve) ** 2
            Az = abs(vg.T @ np.kron(2 * np.array([[1, 0], [0, 1]]), np.identity(2)) @ ve) ** 2

            intensity = (etaX * Ax + etaY * Ay + etaZ * Az) * np.exp(-(Ee - Ee_vals[0]) / (kB * 12))
            I.append(intensity)
    return T, I


# Define the parameters for your simulation
G1 = 1.0     # Spectral splitting between peaks a and d in GHz
G2 = 0.8     # Spectral splitting between peaks b and c in GHz
tg = 0.0     # Ground state splitting angle (radians, 0 means spin-orbit splitting)
pg = np.pi/4 # Ground state angle (free parameter)
te = np.pi/2 # Excited state splitting angle (radians, pi/2 means Jahn-Teller effect)
pe = np.pi/4 # Excited state angle (free parameter)
tB = 30.0    # Angle between high symmetry axis and magnetic field in degrees
f = 0.9      # Factor to diminish orbital g-factor
eps = 0.5    # Uniaxial stress in GPa

# Run the simulation with the specified parameters
results = simulate_model(G1, G2, tg, pg, te, pe, tB, f, eps)

# Access the results
print("Ground State Splitting (DeltaEg):", results['DeltaEg'], "GHz")
print("Excited State Splitting (DeltaEe):", results['DeltaEe'], "GHz")

# Inspect results for magnetic field values
print("Magnetic Field (B-field):", results['B_field'])
print("Energy levels for ground state (Eg) across B-field:", results['Eg'])
print("Energy levels for excited state (Ee) across B-field:", results['Ee'])

# Inspect transition frequencies and intensities
print("Transition frequencies (T):", results['T'])
print("Transition intensities (I):", results['I'])
