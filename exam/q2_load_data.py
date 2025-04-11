from scipy.io import loadmat
import pandas as pd

data = loadmat("Heart_of_Gold_Improp_drive/Heart_of_Gold_Improp_drive.mat")

print(data.keys())

event = data["None"]
evetn_p = data["Events_probability"]
T = 1 #s
u = data["u"]
x_truc = data["x_trunc"]
Z = data["Z"]

event = pd.read_csv("Heart_of_Gold_Improp_drive/event.csv", header=None)
event = event.transpose().reset_index(drop=True)
event.columns = ["event"]

evetn_p = pd.DataFrame(evetn_p).T
evetn_p.columns = ["event_p"]

u = pd.DataFrame(u).T
u.columns = ["a_x", "w_x", "w_y", "w_z"]

x_truc = pd.DataFrame(x_truc).T
x_truc.columns = ["x", "y", "z", "v", "roll", "pitch", "yaw"]

z = pd.DataFrame(Z).T
z.columns = ["x", "y", "z", "roll"]