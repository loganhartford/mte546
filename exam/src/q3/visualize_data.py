import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv("Neural_network_data2/train.csv")

bearing_columns = ['bearing_x', 'bearing_y', 'bearing_z']
nacelle_columns = ['nacelle_x', 'nacelle_y', 'nacelle_z']
label_column = 'label'

fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

for column in bearing_columns:
    axes[0].plot(train[column], label=column)
axes[0].set_title("Bearing Data (x, y, z)")
axes[0].legend()

for column in nacelle_columns:
    axes[1].plot(train[column], label=column)
axes[1].set_title("Nacelle Data (x, y, z)")
axes[1].legend()

axes[2].plot(train[label_column], label=label_column, color='red')
axes[2].set_title(label_column)
axes[2].legend()

plt.tight_layout()
plt.show()
