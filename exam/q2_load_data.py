import matplotlib.pyplot as plt
from scipy.io import loadmat
import pandas as pd

# Load data
data = loadmat("Heart_of_Gold_Improp_drive/Heart_of_Gold_Improp_drive.mat")

# Extract and process data
event = pd.read_csv("Heart_of_Gold_Improp_drive/event.csv", header=None)
event = event.transpose().reset_index(drop=True)
event.columns = ["event"]

evetn_p = pd.DataFrame(data["Events_probability"]).T
evetn_p.columns = ["event_p"]

u = pd.DataFrame(data["u"]).T
u.columns = ["a_x", "w_x", "w_y", "w_z"]

x_truc = pd.DataFrame(data["x_trunc"]).T
x_truc.columns = ["x", "y", "z", "v", "roll", "pitch", "yaw"]

z = pd.DataFrame(data["Z"]).T
z.columns = ["x", "y", "z", "roll"]

# Calculate and print the standard deviation of z for indices 41-70
z_std = z.iloc[41:71].var()
print("Standard deviation of z for indices 41-70:")
print(z_std)

# Plot data
plt.figure(figsize=(12, 10))

# Plot u (e.g., acceleration and angular velocities)
plt.subplot(3, 2, 1)
plt.plot(u["a_x"], label="a_x")
plt.plot(u["w_x"], label="w_x")
plt.plot(u["w_y"], label="w_y")
plt.plot(u["w_z"], label="w_z")
plt.title("u: Acceleration and Angular Velocities")
plt.legend()

# Plot x_truc (e.g., position)
plt.subplot(3, 2, 2)
plt.plot(x_truc["x"], label="x")
plt.plot(x_truc["y"], label="y")
plt.plot(x_truc["z"], label="z")
plt.title("x_truc: Position")
plt.legend()

# Plot z (e.g., truncated position and roll)
plt.subplot(3, 2, 3)
plt.plot(z["x"], label="x")
plt.plot(z["y"], label="y")
plt.plot(z["z"], label="z")
plt.title("z: Truncated Position")
plt.legend()

# Plot event probabilities
plt.subplot(3, 2, 4)
plt.plot(evetn_p["event_p"], label="event_p")
plt.title("Event Probabilities")
plt.legend()

# Plot orientations (roll, pitch, yaw)
plt.subplot(3, 2, 5)
plt.plot(x_truc["roll"], label="roll")
plt.plot(x_truc["pitch"], label="pitch")
plt.plot(x_truc["yaw"], label="yaw")
plt.plot(z["roll"], label="z_roll", linestyle='--')
plt.title("x_truc: Orientations (Roll, Pitch, Yaw)")
plt.legend()

plt.tight_layout()
plt.show()
