#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from ekf import EKF as EKF1
from adaptive_ekf import EKF as EKF2

def load_data(file_path):
    data = loadmat(file_path)
    
    u_raw = data["u"]
    x_trunc_raw = data["x_trunc"]
    Z_raw = data["Z"]
    event_p_raw = data["Events_probability"]
    
    event_csv = "Heart_of_Gold_Improp_drive/event.csv"
    event = pd.read_csv(event_csv, header=None).T.reset_index(drop=True)
    event.columns = ["event"]
    
    u = pd.DataFrame(u_raw).T
    u.columns = ["a_x", "w_x", "w_y", "w_z"]
    
    x_truc = pd.DataFrame(x_trunc_raw).T
    x_truc.columns = ["x", "y", "z", "v", "roll", "pitch", "yaw"]
    
    z = pd.DataFrame(Z_raw).T
    z.columns = ["x", "y", "z", "roll"]
    
    event_p = pd.DataFrame(event_p_raw).T
    event_p.columns = ["event_probability"]

    return u, z, x_truc, event, event_p

def run_ekf1(u, z, x_truc, N=100):
    Q = np.diag([1.0, 1.0, 2.5, 0.5, 0.5, 5.0, 5.0])
    R = np.diag([5000.0, 5000.0, 5000.0, 100.0])

    P0 = Q.copy()
    
    x0 = np.zeros(7)
    dt = 1.0
    ekf = EKF1(P0, Q, R, x0, dt)
    
    x_est_store = np.zeros((N, 7))
    x_est_store[0] = x0
    kalman_gains = []
    errors = []
    for i in range(1, N):
        u_i = u.iloc[i].values.astype(float)
        ekf.predict(u_i)
        
        z_i = z.iloc[i].values.astype(float)
        ekf.update(z_i)
        
        x_est_store[i] = ekf.get_state()
        kalman_gains.append(ekf.K.copy())
        errors.append(x_truc.iloc[i].values - x_est_store[i])
    
    kalman_gains = np.array(kalman_gains)
    errors = np.array(errors)
    return x_est_store, kalman_gains, errors

def run_ekf2(u, z, x_truc, event, event_p, N=100, validation=False):
    Q = np.diag([1.0, 1.0, 2.5, 0.5, 0.5, 0.5, 0.5])
    R = np.diag([500.0, 500.0, 500.0, 1.0])
    
    P0 = Q.copy()
    
    x0 = np.zeros(7)
    dt = 1.0
    ekf = EKF2(P0, Q, R, x0, dt)
    
    x_est_store = np.zeros((N, 7))
    x_est_store[0] = x0
    kalman_gains = []
    errors = []
    for i in range(1, N):
        event_str = event.iloc[i, 0].strip().lower()
        prob_val = event_p.iloc[i, 0]
        
        u_i = u.iloc[i].values.astype(float)
        z_i = z.iloc[i].values.astype(float)
        
        ekf.predict(u_i, event_str, prob_val)
        ekf.update(z_i, event_str, prob_val)
        
        x_est_store[i] = ekf.get_state()
        kalman_gains.append(ekf.K.copy())
        if not validation:
            errors.append(x_truc.iloc[i].values - x_est_store[i])
    kalman_gains = np.array(kalman_gains)
    errors = np.array(errors)

    return x_est_store, kalman_gains, errors

def plot_subranges(t, x1, x2, x_true, z_meas, title_str):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    
    subranges = [(1, 40), (41, 70), (71, 100)]
    titles = ["Samples 1–100", "Samples 1–40", "Samples 41–70", "Samples 71–100"]
    
    ranges = [(0, len(t))] + [(s-1, e) for s, e in subranges]
    for i, (start, end) in enumerate(ranges):
        ax = axs[i // 2, i % 2]
        ax.plot(t[start:end], x1[start:end], 'r-', label='EKF1')
        ax.plot(t[start:end], x2[start:end], 'b-', label='EKF2')
        
        if z_meas is not None:
            ax.plot(t[start:end], z_meas[start:end], 'g-', label='Measurement')
        
        ax.plot(t[start:end], x_true[start:end], 'k--', label='True')
        ax.set_title(f"{title_str} ({titles[i]})")
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(title_str)
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"img/q2/{title_str}_subplots.png")

def compute_statistics(errors_ekf, label, state_indices=[0,3]):
    print(label)
    
    for idx in state_indices:
        err = errors_ekf[:, idx]
        rmse_val = np.sqrt(np.mean(err**2))
        mean_val = np.mean(err)
        std_val = np.std(err)
        print(idx, f"RMSE={rmse_val:.4f}, Mean={mean_val:.4f}, Std={std_val:.4f}")

def compute_gain_norms(kalman_gains):
    if len(kalman_gains.shape) == 3:
        return np.sqrt(np.sum(kalman_gains**2, axis=(1,2)))
    K_reshaped = kalman_gains.reshape(kalman_gains.shape[0], 4, 7)
    return np.sqrt(np.sum(K_reshaped**2, axis=(1,2)))

def plot_gain_norms(norm_ekf1, norm_ekf2):
    t = np.arange(1, len(norm_ekf1)+1)
    
    plt.figure(figsize=(8,4))
    plt.plot(t, norm_ekf1, 'r-', label='EKF1 Gain Norm')
    plt.plot(t, norm_ekf2, 'b-', label='EKF2 Gain Norm')
    plt.xlabel('Time (s)')
    plt.ylabel('||K||')
    plt.title('Kalman Gain Norm (EKF1 vs EKF2)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("img/q2/kalman_gain.png")
    
    subranges = [(1,40), (41,70), (71,len(norm_ekf1))]
    
    print("\nKalman Gain Norms: mean ± std in subranges:")
    for (start, end) in subranges:
        idx1, idx2 = start-1, end
        seg_1 = norm_ekf1[idx1:idx2]
        seg_2 = norm_ekf2[idx1:idx2]
        mean1, std1 = np.mean(seg_1), np.std(seg_1)
        mean2, std2 = np.mean(seg_2), np.std(seg_2)
        
        print(f"{start}-{end} EKF1 mean={mean1:.4f}, std={std1:.4f}; EKF2 mean={mean2:.4f}, std={std2:.4f}")

def main():
    u, z, x_truc, event, event_p = load_data("Heart_of_Gold_Improp_drive/Heart_of_Gold_Improp_drive.mat")

    u.loc[len(u)] = [0.0, 0.0, 0.0, 0.0,]
    
    x_est1, gains1, errors1 = run_ekf1(u, z, x_truc)
    x_est2, gains2, errors2 = run_ekf2(u, z, x_truc, event, event_p)
    
    T = np.arange(100)
    x1_meas = z.iloc[:100, 0].values
    
    plot_subranges(T, x_est1[:,0], x_est2[:,0], x_truc.iloc[:100,0].values, x1_meas, "State X(1)")
    plot_subranges(T, x_est1[:,3], x_est2[:,3], x_truc.iloc[:100,3].values, None, "State X(4) = v")
    
    compute_statistics(errors1, "EKF1", [0,3])
    compute_statistics(errors2, "EKF2", [0,3])
    
    norm1 = compute_gain_norms(gains1)
    norm2 = compute_gain_norms(gains2)
    plot_gain_norms(norm1, norm2)

    x_est2, gains2, errors2 = run_ekf2(u, z, x_truc, event, event_p, N=150, validation=True)
    np.savetxt("ImprobDrive_20736975.csv", x_est2[100:], delimiter=",", fmt="%.6f", header="", comments="")

if __name__ == "__main__":
    main()
