import numpy as np
import h5py
import datetime
from transitions import transitioncompute
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm


def compute_transition_for_point(args):
    """
    Compute transition parameters for a single point in parameter space
    """
    i, j, k, Bx, By, Bz = args

    # Create transition model for this B field configuration
    model = transitioncompute([Bx, By, Bz])

    # Return all computed values to be reassembled later
    return (
        i, j, k,
        model.return_levels(),  # energy levels
        model.return_vectors(),  # eigenvectors
        model.get_c_magnitudes()  # c magnitudes
    )


def compute_transitions(resolution=100):
    # Parameter setup
    B_values = np.linspace(0, 5, resolution)
    theta_values = np.linspace(0, np.pi, resolution)
    phi_values = np.linspace(0, 2 * np.pi, resolution)

    # Create meshgrids
    B_grid, theta_grid, phi_grid = np.meshgrid(B_values, theta_values, phi_values)

    # Precompute Cartesian coordinates
    Bx = B_grid * np.sin(theta_grid) * np.cos(phi_grid)
    By = B_grid * np.sin(theta_grid) * np.sin(phi_grid)
    Bz = B_grid * np.cos(theta_grid)

    # Prepare arguments for parallel computation
    compute_args = [
        (i, j, k, Bx[i, j, k], By[i, j, k], Bz[i, j, k])
        for i in range(len(theta_values))
        for j in range(len(phi_values))
        for k in range(len(B_values))
    ]

    # Parallel computation with progress tracking
    num_cores = max(multiprocessing.cpu_count() - 1, 1)
    results = []

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Submit all tasks
        futures = [executor.submit(compute_transition_for_point, args) for args in compute_args]

        # Track progress with tqdm
        for future in tqdm(as_completed(futures), total=len(futures), desc="Computing Transitions"):
            results.append(future.result())

    # Preallocate result arrays
    energy_ground = np.zeros((len(theta_values), len(phi_values), len(B_values), 4))
    energy_excited = np.zeros((len(theta_values), len(phi_values), len(B_values), 4))
    eigenvectors_ground = np.zeros((len(theta_values), len(phi_values), len(B_values), 4, 4), dtype=complex)
    eigenvectors_excited = np.zeros((len(theta_values), len(phi_values), len(B_values), 4, 4), dtype=complex)
    c_magnitudes = np.zeros_like(B_grid)

    # Populate results
    for i, j, k, levels, vectors, c_mag in results:
        energy_ground[i, j, k, :] = levels[0]
        energy_excited[i, j, k, :] = levels[1]
        eigenvectors_ground[i, j, k, :, :] = vectors[0]
        eigenvectors_excited[i, j, k, :, :] = vectors[1]
        c_magnitudes[i, j, k] = c_mag

    # Save results
    timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M")
    with h5py.File(f'data_{resolution}_{timestamp}.h5', 'w') as f:
        f.create_dataset('B_values', data=B_values)
        f.create_dataset('theta_values', data=theta_values)
        f.create_dataset('phi_values', data=phi_values)
        f.create_dataset('c_magnitudes', data=c_magnitudes)

        grp_ground = f.create_group('ground_state')
        grp_ground.create_dataset('energy_levels', data=energy_ground)
        grp_ground.create_dataset('eigenvectors', data=eigenvectors_ground)

        grp_excited = f.create_group('excited_state')
        grp_excited.create_dataset('energy_levels', data=energy_excited)
        grp_excited.create_dataset('eigenvectors', data=eigenvectors_excited)

    return B_values, theta_values, phi_values


# Usage
if __name__ == '__main__':
    compute_transitions()


