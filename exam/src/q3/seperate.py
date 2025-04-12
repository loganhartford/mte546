import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import os

base_path = 'Neural_network_data2'
train = pd.read_csv(os.path.join(base_path, 'train.csv'))
val = pd.read_csv(os.path.join(base_path, 'validation.csv'))
test = pd.read_csv(os.path.join(base_path, 'test.csv'))

bearing_cols = ['bearing_x', 'bearing_y', 'bearing_z']
nacelle_cols = ['nacelle_x', 'nacelle_y', 'nacelle_z']

X_train_bearing = train[bearing_cols].values
X_val_bearing = val[bearing_cols].values
X_train_nacelle = train[nacelle_cols].values
X_val_nacelle = val[nacelle_cols].values

y_train = to_categorical(train['label'].values)
y_val = to_categorical(val['label'].values)
y_val_labels = np.argmax(y_val, axis=1)

scaler_b = StandardScaler()
X_train_bearing = scaler_b.fit_transform(X_train_bearing)
X_val_bearing = scaler_b.transform(X_val_bearing)

scaler_n = StandardScaler()
X_train_nacelle = scaler_n.fit_transform(X_train_nacelle)
X_val_nacelle = scaler_n.transform(X_val_nacelle)

def build_model(input_dim):
    model = Sequential()
    model.add(Dense(16, activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(0.3))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

early_stopping = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)

model_nacelle = build_model(X_train_nacelle.shape[1])
history_nacelle = model_nacelle.fit(X_train_nacelle, y_train,
                                    validation_data=(X_val_nacelle, y_val),
                                    epochs=200, batch_size=128, verbose=0, callbacks=[early_stopping])

model_bearing = build_model(X_train_bearing.shape[1])
history_bearing = model_bearing.fit(X_train_bearing, y_train,
                                    validation_data=(X_val_bearing, y_val),
                                    epochs=200, batch_size=128, verbose=0, callbacks=[early_stopping])

def plot_learning_curves_side_by_side(history1, title1, history2, title2):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(history1.history['accuracy'], label='Train Acc')
    axes[0].plot(history1.history['val_accuracy'], label='Val Acc')
    axes[0].set_title(title1)
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(history2.history['accuracy'], label='Train Acc')
    axes[1].plot(history2.history['val_accuracy'], label='Val Acc')
    axes[1].set_title(title2)
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    plt.tight_layout()
    os.makedirs('img/q3', exist_ok=True)
    plt.savefig('img/q3/learning_curves_side_by_side.png')

plot_learning_curves_side_by_side(
    history_nacelle, 'ANN1: Nacelle Accelerometer Accuracy',
    history_bearing, 'ANN2: Bearing Accelerometer Accuracy'
)

y_pred_nacelle = model_nacelle.predict(X_val_nacelle)
y_pred_bearing = model_bearing.predict(X_val_bearing)

y_pred_nacelle_labels = np.argmax(y_pred_nacelle, axis=1)
y_pred_bearing_labels = np.argmax(y_pred_bearing, axis=1)

acc_nacelle = accuracy_score(y_val_labels, y_pred_nacelle_labels)
cm_nacelle = confusion_matrix(y_val_labels, y_pred_nacelle_labels)

acc_bearing = accuracy_score(y_val_labels, y_pred_bearing_labels)
cm_bearing = confusion_matrix(y_val_labels, y_pred_bearing_labels)

print(f"\nANN1 (Nacelle) Validation Accuracy: {acc_nacelle:.4f}")
print(f"ANN1 Confusion Matrix:\n{cm_nacelle}")

print(f"\nANN2 (Bearing) Validation Accuracy: {acc_bearing:.4f}")
print(f"ANN2 Confusion Matrix:\n{cm_bearing}")

def plot_confusion_matrices_side_by_side(cm1, title1, cm2, title2, filename):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Healthy', 'Faulty'],
                yticklabels=['Healthy', 'Faulty'], ax=axes[0])
    axes[0].set_title(title1)
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Healthy', 'Faulty'],
                yticklabels=['Healthy', 'Faulty'], ax=axes[1])
    axes[1].set_title(title2)
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    plt.tight_layout()
    os.makedirs('img/q3', exist_ok=True)
    plt.savefig(f'img/q3/{filename}')
    plt.close()

plot_confusion_matrices_side_by_side(
    cm_nacelle, 'ANN1 Confusion Matrix',
    cm_bearing, 'ANN2 Confusion Matrix',
    'confusion_matrices_side_by_side.png'
)
