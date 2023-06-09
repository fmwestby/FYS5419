import numpy as np
from src.QubitClass import One_qubit

def Hamiltonian(lmb):
    # Pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])

    # Identity matrix
    I = np.eye(2) 

    # Parameters
    E1 = 0
    E2 = 4
    V11 = 3
    V22 = -3
    V12 = 0.2
    V21 = 0.2

    # calculate variables
    eps = (E1 + E2) / 2
    omega = (E1 - E2) / 2
    c = (V11 + V22) / 2
    omega_z = (V11 - V22) / 2
    omega_x = V12

    H0 = eps * I + omega * sigma_z
    H1 = c * I + omega_z * sigma_z + omega_x * sigma_x
    
    return H0 + lmb * H1

def prepare_state(theta, phi, target = None):
    """
    Prepares a quantum state for given parameters.

    Parameters
    ----------
    theta, phi : float
        Parameters of the quantum state.
    target : numpy array, optional
        The target state, default is None.

    Returns
    -------
    numpy array
        The prepared quantum state.
    """

    I = np.eye(2)
    
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    
    state = np.array([1, 0])
    
    Rx = np.cos(theta/2) * I - 1j * np.sin(theta/2) * sigma_x
    Ry = np.cos(phi/2) * I - 1j * np.sin(phi/2) * sigma_y
    
    state = Ry @ Rx @ state
    
    if target is not None:
        state = target
    
    return state

def get_energy(angles, lmb, number_shots, target = None):
    """
    Calculates the expected energy for given parameters.

    Parameters
    ----------
    angles : list
        A list of two angles.
    lmb : float
        Lambda parameter of the Hamiltonian.
    number_shots : int
        Number of measurements to be made for the energy calculation.
    target : numpy array, optional
        The target state, default is None.

    Returns
    -------
    float
        The expected energy.
    """

    theta, phi = angles[0], angles[1]

    E1 = 0; E2 = 4; V11 = 3; V22 = -3; V12 = 0.2; V21 = 0.2

    eps = (E1 + E2) / 2; omega = (E1 - E2) / 2; c = (V11 + V22) / 2; omega_z = (V11 - V22) / 2; omega_x = V12

    init_state = prepare_state(theta, phi, target)
    
    qubit = One_qubit()
    qubit.set_state(init_state)
    
    measure_z = qubit.measure(number_shots)

    qubit.set_state(init_state)
    qubit.apply_hadamard()
    
    measure_x = qubit.measure(number_shots)
    
    # expected value of Z = (number of 0 measurements - number of 1 measurements)/ number of shots
    # number of 1 measurements = sum(measure_z)
    exp_val_z = (omega + lmb*omega_z)*(number_shots - 2*np.sum(measure_z)) / number_shots
    exp_val_x = lmb*omega_x*(number_shots - 2*np.sum(measure_x)) / number_shots
    exp_val_i = (eps + c*lmb) 
    
    exp_val = (exp_val_z + exp_val_x + exp_val_i)
    
    return exp_val 



def minimize_energy(lmb, number_shots, angles_0, learning_rate, max_epochs):
    """
    Minimizes the expected energy of a quantum state.

    Parameters
    ----------
    lmb : float
        Lambda parameter of the Hamiltonian.
    number_shots : int
        Number of measurements to be made for the energy calculation.
    angles_0 : list
        A list of two initial angles.
    learning_rate : float
        Learning rate for the gradient descent optimization.
    max_epochs : int
        Maximum number of iterations for the optimization.

    Returns
    -------
    tuple
        A tuple containing final angles, number of epochs taken, a boolean indicating
        whether the optimization has converged, final energy, and energy change in the
        last iteration.
    """
    
    angles = angles_0 #lmb*np.array([np.pi, np.pi])
    epoch = 0
    delta_energy = 1
    
    energy = get_energy(angles, lmb, number_shots)
    
    while (epoch < max_epochs) and (delta_energy > 1e-4):
        grad = np.zeros_like(angles)
        
        for idx in range(angles.shape[0]):
            angles_temp = angles.copy()
            angles_temp[idx] += np.pi/2 
            E_plus = get_energy(angles_temp, lmb, number_shots)
            angles_temp[idx] -= np.pi 
            E_minus = get_energy(angles_temp, lmb, number_shots)
            grad[idx] = (E_plus - E_minus)/2 
        
        angles -= learning_rate*grad 
        new_energy = get_energy(angles, lmb, number_shots)
        delta_energy = np.abs(new_energy - energy)
        
        energy = new_energy
        
        epoch += 1
    
    return angles, epoch, (epoch < max_epochs), energy, delta_energy