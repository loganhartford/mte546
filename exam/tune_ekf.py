#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize
from ekf import EKF

def run_ekf_tuning(q_diag, r_diag, u, z, x_truc):
    """
    Runs the EKF simulation for N=100 steps using the specified Q and R diagonal values.
    
    Parameters:
        q_diag: list/array of 7 process noise diagonal values.
        r_diag: list/array of 4 measurement noise diagonal values.
        u     : DataFrame of control inputs (columns: ["a_x", "w_x", "w_y", "w_z"]).
        z     : DataFrame of measurements (columns: ["x", "y", "z", "roll"]).
        x_truc: DataFrame of ground truth states 
                (columns: ["x", "y", "z", "v", "roll", "pitch", "yaw"]).
        
    Returns:
        x_est_store: Array (N x 7) containing the estimated state at each timestep.
    """
    Q = np.diag(q_diag)
    R = np.diag(r_diag)
    P0 = Q.copy()
    dt = 1.0
    # Initialize state using the first ground truth row for better sensitivity.
    x0 = x_truc.iloc[0, :7].values.astype(float)
    
    ekf = EKF(P0, Q, R, x0, dt)
    N = 100
    x_est_store = np.zeros((N, 7), dtype=float)
    x_est_store[0, :] = x0

    for i in range(1, N):
        u_i = u.iloc[i].values.astype(float)
        ekf.predict(u_i)
        z_i = z.iloc[i].values.astype(float)
        ekf.update(z_i)
        x_est_store[i, :] = ekf.get_state()
    
    return x_est_store

def objective_function_log(log_params, u, z, x_truc):
    """
    Objective function computed in log-space.
    
    Parameters:
        log_params: array of length 11 containing log(Q_diag) and log(R_diag).
        u, z, x_truc: DataFrames of controls, measurements, and ground truth.
    
    Returns:
        mse: Mean squared error over all 7 state components.
    """
    # Convert log parameters back to actual values.
    params = np.exp(log_params)
    q_diag = params[:7]
    r_diag = params[7:]
    
    # Run EKF tuning simulation.
    x_est = run_ekf_tuning(q_diag, r_diag, u, z, x_truc)
    gt_state = x_truc.iloc[:100, :7].values  # ground truth for full state.
    mse = np.mean(np.sum((x_est - gt_state)**2, axis=1))
    return mse

def main():
    # Load data.
    data = loadmat("Heart_of_Gold_Improp_drive/Heart_of_Gold_Improp_drive.mat")
    u_raw = data["u"]
    x_trunc_raw = data["x_trunc"]
    Z_raw = data["Z"]
    
    u = pd.DataFrame(u_raw).T
    u.columns = ["a_x", "w_x", "w_y", "w_z"]
    
    x_truc = pd.DataFrame(x_trunc_raw).T
    x_truc.columns = ["x", "y", "z", "v", "roll", "pitch", "yaw"]
    
    z = pd.DataFrame(Z_raw).T
    z.columns = ["x", "y", "z", "roll"]

    # Initial guess for Q and R diagonal values.
    initial_q = [1.0, 1.0, 2.5, 0.5, 0.5, 0.5, 0.5]
    initial_r = [500.0, 500.0, 500.0, 1.0]
    initial_guess = np.array(initial_q + initial_r)
    
    # Optimize in log-space.
    log_initial_guess = np.log(initial_guess)
    bounds = [(-20, None)] * (7 + 4)
    
    result = minimize(objective_function_log, log_initial_guess, args=(u, z, x_truc),
                      bounds=bounds, method='L-BFGS-B')
    
    optimal_params = np.exp(result.x)
    print("Optimal Q diagonal values:")
    print(optimal_params[:7])
    print("Optimal R diagonal values:")
    print(optimal_params[7:])
    print("Objective function (MSE) value:", result.fun)
    
    # Run EKF with optimal Q and R and plot full state.
    optimal_x_est = run_ekf_tuning(optimal_params[:7], optimal_params[7:], u, z, x_truc)
    t = np.arange(100)
    state_labels = ["x", "y", "z", "v", "roll", "pitch", "yaw"]
    fig, axs = plt.subplots(4, 2, figsize=(12, 15))
    axs = axs.flatten()
    
    for i in range(7):
        axs[i].plot(t, optimal_x_est[:, i], 'r-', label='EKF estimate')
        axs[i].plot(t, x_truc.iloc[:100, i].values, 'k--', label='Actual')
        axs[i].set_ylabel(state_labels[i])
        axs[i].set_xlabel("Time step")
        axs[i].legend(loc="best")
        axs[i].grid(True)
    
    # Remove any unused subplot.
    fig.delaxes(axs[-1])
    
    plt.tight_layout()
    plt.savefig("tuned_full_state.png")
    plt.show()

if __name__ == "__main__":
    main()
