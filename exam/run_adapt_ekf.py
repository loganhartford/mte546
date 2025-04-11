#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat

from ekf import EKF

def run_ekf_adaptive(u, z, x_truc, event, event_p, adaptive_factor=5.0, beta=1e-4):
    
    Q_base = np.diag([1.0, 1.0, 2.5, 0.5, 0.5, 0.5, 0.5])
    R_base = np.diag([500.0, 500.0, 500.0, 1.0])

    Q_base = np.diag([1.0, 1.0, 2.5, 0.5, 0.5, 5.0, 5.0])
    R_base = np.diag([5000.0, 5000.0, 5000.0, 100.0])
    
    P0 = Q_base.copy()
    dt = 1.0
    x0 = np.zeros(7, dtype=float)

    ekf = EKF(P0, Q_base.copy(), R_base.copy(), x0, dt)
    
    N = 100
    x_est_store = np.zeros((N, 7), dtype=float)
    x_est_store[0, :] = x0

    for i in range(1, N):
        u_i = u.iloc[i].values.astype(float)
        z_i = z.iloc[i].values.astype(float)
        
        current_event = event.iloc[i, 0].strip().lower()
        if current_event != "usual stuff":
            event_scale = adaptive_factor
            print(i, current_event, event_p.iloc[i, 0])
        else:
            event_scale = 1.0

        meas_norm = np.linalg.norm(z_i)
        meas_scale = 1 + beta * meas_norm

        Q_new = Q_base * event_scale
        R_new = R_base * event_scale * meas_scale

        ekf.Q = Q_new
        ekf.R = R_new

        ekf.predict(u_i)
        ekf.update(z_i)
        x_est_store[i, :] = ekf.get_state()
    
    rmse = np.sqrt(np.mean((x_est_store - x_truc.iloc[:N, :].values)**2, axis=0))
    print("Adaptive EKF RMSE error for each state:")
    print("x, y, z, v, roll, pitch, yaw")
    print(rmse)
    
    return x_est_store

def main():
    data = loadmat("Heart_of_Gold_Improp_drive/Heart_of_Gold_Improp_drive.mat")
    u_raw = data["u"]
    x_trunc_raw = data["x_trunc"]
    Z_raw = data["Z"]
    event_p = data["Events_probability"]
    
    event = pd.read_csv("Heart_of_Gold_Improp_drive/event.csv", header=None).T.reset_index(drop=True)
    event.columns = ["event"]
    
    u = pd.DataFrame(u_raw).T
    u.columns = ["a_x", "w_x", "w_y", "w_z"]
    
    x_truc = pd.DataFrame(x_trunc_raw).T
    x_truc.columns = ["x", "y", "z", "v", "roll", "pitch", "yaw"]
    
    z = pd.DataFrame(Z_raw).T
    z.columns = ["x", "y", "z", "roll"]

    event_p = pd.DataFrame(event_p).T
    event_p.columns = ["event_probability"]

    x_est_store = run_ekf_adaptive(u, z, x_truc, event, event_p, adaptive_factor=5.0)
    t = np.arange(100)
    
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10), sharex=True)
    
    axs[0, 0].plot(t, x_est_store[:, 0], 'r-', label='EKF estimate')
    axs[0, 0].plot(t, x_truc.iloc[:100, 0], 'k--', label='Actual')
    axs[0, 0].set_ylabel('x (m)')
    axs[0, 0].set_title('Position X')
    axs[0, 0].legend(loc='best')
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
    
    angles = ['roll', 'pitch', 'yaw']
    for i, label in enumerate(angles):
        axs[1, i].plot(t, x_est_store[:, 4 + i], 'r-', label='EKF estimate')
        axs[1, i].plot(t, x_truc.iloc[:100, 4 + i], 'k--', label='Actual')
        axs[1, i].set_ylabel(label)
        axs[1, i].set_title(f'Orientation {label.capitalize()}')
        axs[1, i].legend(loc='best')
        axs[1, i].grid(True)
        
    axs[1, 0].set_xlabel('Time step')
    axs[1, 1].set_xlabel('Time step')
    axs[1, 2].set_xlabel('Time step')
    fig.tight_layout()
    plt.savefig('pos_orien_adaptive.png')
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(8,4))
    ax2.plot(t, x_est_store[:, 3], 'r-', label='EKF estimate')
    ax2.plot(t, x_truc.iloc[:100, 3], 'k--', label='Actual')
    ax2.set_ylabel('v (m/s)')
    ax2.set_xlabel('Time step')
    ax2.legend(loc='best')
    ax2.grid(True)
    ax2.set_title('Forward Velocity')
    plt.tight_layout()
    plt.savefig('vel_adaptive.png')
    plt.show()

if __name__ == "__main__":
    main()
