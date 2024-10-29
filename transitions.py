import numpy as np
import matplotlib.pyplot as plt

def return_transitions(Ee, Eg):
    '''
    :param Ee: eigenvalues of excited hamiltonian
    :param Eg: eigenvectors of ground hamiltonian
    :return: numpy array of all 16 transitions
    '''
    transitions = []
    for e in Ee:
        for g in Eg:
            transitions.append(e-g)
    return np.array(transitions)

def return_field_amp(Ve, Vg):
    '''
    :param Ve: excited state eigenvector
    :param Vg: ground state eigenvector
    :return: field amplitudes
    '''
    vg = Vg / np.linalg.norm(Vg)
    ve = Ve / np.linalg.norm(Ve)


    Px = np.kron(np.array([[1, 0], [0, -1]]), np.identity(2))
    Py = np.kron(np.array([[0, -1], [-1, 0]]), np.identity(2))
    Pz = np.kron(2 * np.array([[1, 0], [0, 1]]), np.identity(2))

    Ax = np.conj(vg) @ Px @ ve
    Ay = np.conj(vg) @ Py @ ve
    Az = np.conj(vg) @ Pz @ ve

    return Ax, Ay, Az


def convert_lab_frame(Ax, Ay, Az, theta = 54.7*180/np.pi, phi = 0):
    """
    Calculate transformed coordinates using the given matrix multiplication.

    Parameters:
    Ax, Ay, Az (float): field amplitudes
    theta (float): Angle theta in radians
    phi (float): Angle phi in radians

    Returns:
    tuple: Transformed coordinates (Ax_transformed, Ay_transformed)
    """

    M1 = np.array([[1, 0, 0],
                   [0, 1, 0]])

    M2 = np.array([[np.cos(theta), 0, np.sin(theta)],
                   [0, 1, 0],
                   [-np.sin(theta), 0, np.cos(theta)]])

    M3 = np.array([[np.cos(phi), -np.sin(phi), 0],
                   [np.sin(phi), np.cos(phi), 0],
                   [0, 0, 1]])


    A = np.array([Ax, Ay, Az])

    # Perform the matrix multiplication
    result = M1 @ M2 @ M3 @ A

    # Return the transformed coordinates
    return result[0], result[1]


def calculate_final_field(Ax_l, Ay_l, phi):
    """
    Calculate final coordinates using the given matrix multiplication.

    Parameters:
    Ax_l, Ay_l (float): lab frame field
    phi (float): half-wave plate angle

    Returns:
    tuple: Final coordinates (Ax_f, Ay_f)
    """
    # First matrix [1 0; 0 0]
    M1 = np.array([[1, 0],
                   [0, 0]])

    # Second matrix [cos(2φ) sin(2φ); sin(2φ) -cos(2φ)]
    M2 = np.array([[np.cos(2 * phi), np.sin(2 * phi)],
                   [np.sin(2 * phi), -np.cos(2 * phi)]])

    # Input vector
    A_l = np.array([Ax_l, Ay_l])

    # Perform the matrix multiplication
    result = M1 @ M2 @ A_l

    # Return the final coordinates
    return result[0], result[1]
def scan_polarisation(Ax, Ay, Az):
    phi_values = np.linspace(0, 2 * np.pi, 360)
    magnitudes = []

    # Calculate magnitude for each phi
    for phi in phi_values:
        # First transformation
        Ax_l, Ay_l = convert_lab_frame(Ax, Ay, Az)

        # Second transformation
        Ax_f, Ay_f = calculate_final_field(Ax_l, Ay_l, phi)

        # Calculate magnitude
        magnitude = np.sqrt(np.abs(Ax_f) ** 2 + np.abs(Ay_f) ** 2)
        magnitudes.append(magnitude)
    return phi_values, magnitudes

def get_magnitude_from_vector(Ve, Vg):
    Ax, Ay, Az = return_field_amp(Ve, Vg)
    return scan_polarisation(Ax, Ay, Az)

def plot_magnitude_polar(phi_values, magnitudes):
    """
    Creates a polar plot of the magnitude of the final vector (sqrt(Ax_f^2 + Ay_f^2))
    as a function of phi.
    """

    # Create polar plot
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, projection='polar')

    # Plot magnitude vs phi
    ax.plot(phi_values, magnitudes)

    # Customize the plot
    ax.set_title('Magnitude of Final Vector vs φ')
    ax.grid(True)
    plt.show()

x,y = 2+3j, 1.5+2j
print(x**2+y**2)
print(np.sqrt(x**2+y**2))