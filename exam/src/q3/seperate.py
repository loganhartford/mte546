import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

# Load datasets
base_path = 'Neural_network_data2'
train = pd.read_csv(os.path.join(base_path, 'train.csv'))
val = pd.read_csv(os.path.join(base_path, 'validation.csv'))
test = pd.read_csv(os.path.join(base_path, 'test.csv'))

# Separate inputs
bearing_cols = ['bearing_x', 'bearing_y', 'bearing_z']
nacelle_cols = ['nacelle_x', 'nacelle_y', 'nacelle_z']

X_train_bearing = train[bearing_cols].values
X_val_bearing = val[bearing_cols].values

X_train_nacelle = train[nacelle_cols].values
X_val_nacelle = val[nacelle_cols].values

y_train = to_categorical(train['label'].values)
y_val = to_categorical(val['label'].values)

# Normalize inputs
scaler_b = StandardScaler()
X_train_bearing = scaler_b.fit_transform(X_train_bearing)
X_val_bearing = scaler_b.transform(X_val_bearing)

scaler_n = StandardScaler()
X_train_nacelle = scaler_n.fit_transform(X_train_nacelle)
X_val_nacelle = scaler_n.transform(X_val_nacelle)

# Create model builder function
def build_model(input_dim):
    model = Sequential()
    model.add(Dense(16, activation='relu', input_shape=(input_dim,)))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Train ANN1 (nacelle)
model_nacelle = build_model(X_train_nacelle.shape[1])
history_nacelle = model_nacelle.fit(X_train_nacelle, y_train,
                                    validation_data=(X_val_nacelle, y_val),
                                    epochs=50, batch_size=32, verbose=0)

# Train ANN2 (bearing)
model_bearing = build_model(X_train_bearing.shape[1])
history_bearing = model_bearing.fit(X_train_bearing, y_train,
                                    validation_data=(X_val_bearing, y_val),
                                    epochs=50, batch_size=32, verbose=0)

# Plot learning curves
def plot_learning_curves(history, title):
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_learning_curves(history_nacelle, 'ANN1: Nacelle Accelerometer Accuracy')
plot_learning_curves(history_bearing, 'ANN2: Bearing Accelerometer Accuracy')
