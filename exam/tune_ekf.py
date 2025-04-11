#!/usr/bin/env python3
"""
optimize_QR_scaled.py

This script optimizes the diagonal elements of the process noise covariance Q
and measurement noise covariance R used in an Extended Kalman Filter (EKF) while
accounting for scaling differences across state variables.

Usage:
    python optimize_QR_scaled.py

Modifications compared to the earlier script:
  - A weight_vector is introduced to scale the errors for each state.
  - The weighted error metric is used in the objective so that state errors with
    large nominal scales (e.g., positions ~1e6) don’t dominate those with smaller
    scales (e.g., orientations ~1e1).

Make sure you have the following:
  - The data files: "Heart_of_Gold_Improp_drive/Heart_of_Gold_Improp_drive.mat" and "Heart_of_Gold_Improp_drive/event.csv"
  - An EKF implementation in ekf.py with methods: predict(u), update(z), and get_state().
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize

# Import the EKF class (make sure the ekf.py module is available)
from ekf import EKF

def load_data():
    """
    Loads the dataset used in the EKF simulation.
    Returns:
        u (pd.DataFrame): Control inputs.
        z (pd.DataFrame): Sensor measurements.
        x_truc (pd.DataFrame): Ground truth states (first 100 samples for tuning).
    """
    data = loadmat("Heart_of_Gold_Improp_drive/Heart_of_Gold_Improp_drive.mat")
    event = pd.read_csv("Heart_of_Gold_Improp_drive/event.csv", header=None).T.reset_index(drop=True)
    event.columns = ["event"]

    # Convert arrays to DataFrames (transpose to get appropriate shape)
    u = pd.DataFrame(data["u"]).T
    u.columns = ["a_x", "w_x", "w_y", "w_z"]

    x_truc = pd.DataFrame(data["x_trunc"]).T
    x_truc.columns = ["x", "y", "z", "v", "roll", "pitch", "yaw"]

    z = pd.DataFrame(data["Z"]).T
    z.columns = ["x", "y", "z", "roll"]

    return u, z, x_truc

def simulate_ekf(u, z, x_truc, Q_diag, R_diag, weight_vector=None):
    """
    Simulates the EKF over 100 time steps with a given Q and R.
    Args:
        u (pd.DataFrame): Control input dataframe.
        z (pd.DataFrame): Sensor measurement dataframe.
        x_truc (pd.DataFrame): Ground truth states.
        Q_diag (list or array): Diagonal entries for Q (length=7).
        R_diag (list or array): Diagonal entries for R (length=4).
        weight_vector (np.ndarray): A 1D array of weights for each state variable 
                                    (length=7) to balance the error terms.
    Returns:
        error (float): The weighted error metric.
        x_est_store (np.ndarray): The estimated states over 100 time steps.
    """
    Q = np.diag(Q_diag)
    R = np.diag(R_diag)
    P0 = Q.copy()
    x0 = np.zeros(7, dtype=float)
    dt = 1.0

    ekf = EKF(P0, Q, R, x0, dt)
    N = 100
    x_est_store = np.zeros((N, 7), dtype=float)

    for i in range(1, N):
        u_i = u.iloc[i].values.astype(float)
        ekf.predict(u_i)
        z_i = z.iloc[i].values.astype(float)
        ekf.update(z_i)
        x_est_store[i, :] = ekf.get_state()

    true_states = x_truc.iloc[:N].values.astype(float)
    residual = x_est_store - true_states

    # If a weight_vector is provided, scale the residuals elementwise.
    # Ensure weight_vector is a (7,) shape array.
    if weight_vector is not None:
        residual = residual * weight_vector  # Elementwise multiplication

    error = np.linalg.norm(residual)
    return error, x_est_store

def objective(params, u, z, x_truc, weight_vector):
    """
    Objective function for the optimizer.
    Args:
        params (np.ndarray): Array of 11 parameters where:
            - params[0:7] are Q diagonal elements.
            - params[7:11] are R diagonal elements.
        u, z, x_truc: Dataframes containing controls, sensor data, and true states.
        weight_vector (np.ndarray): Weight vector for scaling the error metric.
    Returns:
        float: The weighted error metric to minimize.
    """
    Q_diag = params[0:7]
    R_diag = params[7:11]
    error, _ = simulate_ekf(u, z, x_truc, Q_diag, R_diag, weight_vector)
    return error

def main():
    # Load data from file
    u, z, x_truc = load_data()

    # Define weight_vector for the states.
    # Adjust these factors based on the expected scale of each state variable.
    # For example, if positions (x, y, z) are ~1e6 in magnitude, use 1/1e6.
    # If orientations (roll, pitch, yaw) are ~1e1, use 1/1e1.
    # Here we assume:
    #   - positions: indices 0,1,2 => weight factor 1/1e6
    #   - velocity: index 3        => weight factor 1 (adjust if needed)
    #   - orientation: indices 4,5,6 => weight factor 1/10
    weight_vector = np.array([1e-6, 1e-6, 1e-6, 1.0, 1e-1, 1e-1, 1e-1])

    # Initial guess for Q and R diagonals (same as original settings)
    initial_Q = [1.0, 1.0, 2.5, 0.5, 0.5, 5.0, 5.0]
    initial_R = [5000.0, 5000.0, 5000.0, 100.0]
    initial_params = np.array(initial_Q + initial_R)

    # Set bounds for the parameters (ensuring they remain positive)
    bounds = [(1e-6, None)] * 11

    print("Starting optimization of Q and R matrices with scaling...")

    # Use L-BFGS-B to minimize the weighted error metric
    result = minimize(objective, initial_params, args=(u, z, x_truc, weight_vector),
                      bounds=bounds, method='L-BFGS-B',
                      options={'disp': True, 'maxiter': 200})
    print("\nOptimization Result:")
    print(result)

    # Retrieve the optimized Q and R diagonal elements
    best_params = result.x
    best_Q = best_params[0:7]
    best_R = best_params[7:11]
    print("\nOptimized Q diagonal: ", best_Q)
    print("Optimized R diagonal: ", best_R)

    # Final simulation with optimized parameters
    final_error, x_est_store = simulate_ekf(u, z, x_truc, best_Q, best_R, weight_vector)
    print("Final weighted simulation error: {:.4f}".format(final_error))

    # Plot the estimated states against the true states
    t = np.arange(100)
    fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharex=True)
    
    # Position plots (x, y, z)
    axs[0, 0].plot(t, x_est_store[:, 0], 'r-', label='EKF estimate')
    axs[0, 0].plot(t, x_truc.iloc[:100, 0], 'k--', label='Actual')
    axs[0, 0].set_ylabel('x (m)')
    axs[0, 0].set_title('Position X')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    axs[0, 1].plot(t, x_est_store[:, 1], 'r-', label='EKF estimate')
    axs[0, 1].plot(t, x_truc.iloc[:100, 1], 'k--', label='Actual')
    axs[0, 1].set_ylabel('y (m)')
    axs[0, 1].set_title('Position Y')
    axs[0, 1].grid(True)

    axs[0, 2].plot(t, x_est_store[:, 2], 'r-', label='EKF estimate')
    axs[0, 2].plot(t, x_truc.iloc[:100, 2], 'k--', label='Actual')
    axs[0, 2].set_ylabel('z (m)')
    axs[0, 2].set_title('Position Z')
    axs[0, 2].grid(True)

    # Orientation plots (roll, pitch, yaw)
    angles = ['roll', 'pitch', 'yaw']
    for i, label in enumerate(angles):
        axs[1, i].plot(t, x_est_store[:, 4+i], 'r-', label='EKF estimate')
        axs[1, i].plot(t, x_truc.iloc[:100, 4+i], 'k--', label='Actual')
        axs[1, i].set_ylabel(label)
        axs[1, i].set_title(f'Orientation {label.capitalize()}')
        axs[1, i].grid(True)
        axs[1, i].legend()

    for ax in axs[1, :]:
        ax.set_xlabel('Time Step')

    fig.tight_layout()
    plt.savefig('optimized_pos_orien_scaled.png')
    plt.show()

if __name__ == "__main__":
    main()
