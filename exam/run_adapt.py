import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from adaptive_ekf import EKF

def run_adaptive_ekf(u, z, x_truc):
    Q = np.diag([1.0, 1.0, 2.5, 0.5, 0.5, 0.5, 0.5])
    R = np.diag([500.0, 500.0, 500.0, 1.0])

    P0 = Q.copy()

    x0 = np.array([
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ], dtype=float)

    dt = 1.0

    ekf = EKF(P0, Q, R, x0, dt)

    N = 100
    x_est_store = np.zeros((N, 7), dtype=float)
    kalman_gains = []
    estimation_errors = []
    prob = 1
    for i in range(1, N):
        current_event = event.iloc[i, 0].strip().lower()
        if current_event != "usual stuff":
            prob = event_p.iloc[i, 0]
        else:
            prob = 1
        
        u_i = u.iloc[i].values.astype(float)
        ekf.predict(u_i, prob)
        z_i = z.iloc[i].values.astype(float)
        ekf.update(z_i, prob)

        kalman_gains.append(ekf.K)
        estimation_errors.append(x_truc.iloc[i].values - ekf.get_state())

        x_est_store[i, :] = ekf.get_state()

    kalman_gains = np.array(kalman_gains).reshape(99, -1)
    print(kalman_gains[0])
    kalman_gains_df = pd.DataFrame(kalman_gains)
    kalman_gains_df.to_csv("kalman_gains_ekf2.csv", index=False)

    estimation_errors_df = pd.DataFrame(estimation_errors, columns=["x", "y", "z", "v", "roll", "pitch", "yaw"])
    estimation_errors_df.to_csv("state_estimation_errors_ekf2.csv", index=False)

    rmse = np.sqrt(np.mean((x_est_store - x_truc.iloc[:100, :].values)**2, axis=0))
    print("RMSE error for each state:")
    print("x, y, z, v, roll, pitch, yaw")
    print(rmse)

    t_100 = np.arange(100)

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10), sharex=True)
    
    axs[0, 0].plot(t_100, x_est_store[:, 0], 'r-', label='EKF estimate')
    axs[0, 0].plot(t_100, x_truc.iloc[:100, 0], 'k--', label='Actual')
    axs[0, 0].set_ylabel('x (m)')
    axs[0, 0].set_title('Position X')
    axs[0, 0].legend(loc='best')
    axs[0, 0].grid(True)

    axs[0, 1].plot(t_100, x_est_store[:, 1], 'r-', label='EKF estimate')
    axs[0, 1].plot(t_100, x_truc.iloc[:100, 1], 'k--', label='Actual')
    axs[0, 1].set_ylabel('y (m)')
    axs[0, 1].set_title('Position Y')
    axs[0, 1].grid(True)

    axs[0, 2].plot(t_100, x_est_store[:, 2], 'r-', label='EKF estimate')
    axs[0, 2].plot(t_100, x_truc.iloc[:100, 2], 'k--', label='Actual')
    axs[0, 2].set_ylabel('z (m)')
    axs[0, 2].set_title('Position Z')
    axs[0, 2].grid(True)

    angles = ['roll', 'pitch', 'yaw']
    for i, label in enumerate(angles):
        axs[1, i].plot(t_100, x_est_store[:, 4 + i], 'r-', label='EKF estimate')
        axs[1, i].plot(t_100, x_truc.iloc[:100, 4 + i], 'k--', label='Actual')
        axs[1, i].set_ylabel(label)
        axs[1, i].set_title(f'Orientation {label.capitalize()}')
        axs[1, i].grid(True)
        axs[1, i].legend(loc='best')

    axs[1, 0].set_xlabel('Time step')
    axs[1, 1].set_xlabel('Time step')
    axs[1, 2].set_xlabel('Time step')
    fig.tight_layout()
    plt.savefig('pos_orien_2.png')
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(t_100, x_est_store[:, 3], 'r-', label='EKF estimate')
    ax2.plot(t_100, x_truc.iloc[:100, 3], 'k--', label='Actual')
    ax2.set_ylabel('v (m/s)')
    ax2.set_xlabel('Time step')
    ax2.grid(True)
    ax2.legend(loc='best')
    ax2.set_title('Forward Velocity')
    fig2.tight_layout()
    plt.savefig('vel_2.png')


if __name__ == "__main__":
    from scipy.io import loadmat

    data = loadmat("Heart_of_Gold_Improp_drive/Heart_of_Gold_Improp_drive.mat")

    event = pd.read_csv("Heart_of_Gold_Improp_drive/event.csv", header=None).T.reset_index(drop=True)
    event.columns = ["event"]

    u = data["u"]
    x_truc = data["x_trunc"]
    Z = data["Z"]
    event_p = data["Events_probability"]

    u = pd.DataFrame(u).T
    u.columns = ["a_x", "w_x", "w_y", "w_z"]

    x_truc = pd.DataFrame(x_truc).T
    x_truc.columns = ["x", "y", "z", "v", "roll", "pitch", "yaw"]

    z = pd.DataFrame(Z).T
    z.columns = ["x", "y", "z", "roll"]

    event_p = pd.DataFrame(event_p).T
    event_p.columns = ["event_probability"]

    run_ekf(u, z, x_truc)
