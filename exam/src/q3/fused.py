import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback
import matplotlib.pyplot as plt
import seaborn as sns
import os

base_path = 'Neural_network_data2'
train = pd.read_csv(os.path.join(base_path, 'train.csv'))
val = pd.read_csv(os.path.join(base_path, 'validation.csv'))

bearing_cols = ['bearing_x', 'bearing_y', 'bearing_z']
nacelle_cols = ['nacelle_x', 'nacelle_y', 'nacelle_z']

X_train_bearing = train[bearing_cols].values
X_val_bearing = val[bearing_cols].values
X_train_nacelle = train[nacelle_cols].values
X_val_nacelle = val[nacelle_cols].values

y_train = to_categorical(train['label'].values)
y_val = to_categorical(val['label'].values)
y_train_labels = train['label'].values
y_val_labels = val['label'].values

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

def build_deep_model(input_dim):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

class LateFusionPredictionLogger(Callback):
    def __init__(self):
        self.train_preds = []
        self.val_preds = []

    def __init__(self, X_train, X_val):
        self.X_train = X_train
        self.X_val = X_val
        self.train_preds = []
        self.val_preds = []

    def on_epoch_end(self, epoch, logs=None):
        self.train_preds.append(self.model.predict(self.X_train, verbose=0))
        self.val_preds.append(self.model.predict(self.X_val, verbose=0))

early_stopping = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)

logger_nacelle = LateFusionPredictionLogger(X_train_nacelle, X_val_nacelle)
logger_bearing = LateFusionPredictionLogger(X_train_bearing, X_val_bearing)

model_nacelle = build_model(X_train_nacelle.shape[1])
model_bearing = build_model(X_train_bearing.shape[1])

history_nacelle = model_nacelle.fit(X_train_nacelle, y_train,
                                    validation_data=(X_val_nacelle, y_val),
                                    epochs=200, batch_size=128, verbose=0,
                                    callbacks=[early_stopping, logger_nacelle])

history_bearing = model_bearing.fit(X_train_bearing, y_train,
                                    validation_data=(X_val_bearing, y_val),
                                    epochs=200, batch_size=128, verbose=0,
                                    callbacks=[early_stopping, logger_bearing])

X_train_fused = np.hstack((X_train_nacelle, X_train_bearing))
X_val_fused = np.hstack((X_val_nacelle, X_val_bearing))

model_fusion = build_deep_model(X_train_fused.shape[1])
history_fusion = model_fusion.fit(X_train_fused, y_train,
                                  validation_data=(X_val_fused, y_val),
                                  epochs=200, batch_size=128, verbose=0,
                                  callbacks=[early_stopping])

def compute_late_fusion_accuracies(train_preds1, train_preds2, y_train_labels,
                                   val_preds1, val_preds2, y_val_labels):
    n_epochs = min(len(train_preds1), len(train_preds2))
    train_accs, val_accs = [], []
    for i in range(n_epochs):
        p_train = train_preds1[i] * train_preds2[i]
        p_val = val_preds1[i] * val_preds2[i]
        train_accs.append(accuracy_score(y_train_labels, np.argmax(p_train, axis=1)))
        val_accs.append(accuracy_score(y_val_labels, np.argmax(p_val, axis=1)))
    return train_accs, val_accs

late_train_acc, late_val_acc = compute_late_fusion_accuracies(
    logger_nacelle.train_preds, logger_bearing.train_preds, y_train_labels,
    logger_nacelle.val_preds, logger_bearing.val_preds, y_val_labels
)

def plot_fusion_learning_curves(history_early, late_train_acc, late_val_acc):
    n_epochs = min(len(history_early.history['accuracy']),
                   len(late_train_acc), len(late_val_acc))

    epochs = range(1, n_epochs + 1)
    early_train_acc = history_early.history['accuracy'][:n_epochs]
    early_val_acc = history_early.history['val_accuracy'][:n_epochs]
    late_train_acc = late_train_acc[:n_epochs]
    late_val_acc = late_val_acc[:n_epochs]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, early_train_acc, label='Train Acc')
    axes[0].plot(epochs, early_val_acc, label='Val Acc')
    axes[0].set_title('Early Fusion (ANN3)')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, late_train_acc, label='Train Acc (Late Fusion)')
    axes[1].plot(epochs, late_val_acc, label='Val Acc (Late Fusion)')
    axes[1].set_title('Late Fusion (Product T-norm)')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    os.makedirs('img/q3', exist_ok=True)
    plt.savefig('img/q3/fusion_learning_curves.png')
    plt.close()

plot_fusion_learning_curves(history_fusion, late_train_acc, late_val_acc)

# Final predictions + confusion matrices
final_y_pred_early = np.argmax(model_fusion.predict(X_val_fused), axis=1)
final_y_pred_late = np.argmax(logger_nacelle.val_preds[-1] * logger_bearing.val_preds[-1], axis=1)

def print_fusion_results(name, y_pred_labels):
    acc = accuracy_score(y_val_labels, y_pred_labels)
    cm = confusion_matrix(y_val_labels, y_pred_labels)
    print(f"\n{name} Fusion Accuracy: {acc:.4f}")
    print(f"{name} Confusion Matrix:\n{cm}")
    return acc, cm

acc_early, cm_early = print_fusion_results("Early", final_y_pred_early)
acc_late, cm_late = print_fusion_results("Late", final_y_pred_late)

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
    cm_early, 'Early Fusion Confusion Matrix',
    cm_late, 'Late Fusion Confusion Matrix',
    'fusion_confusion_matrices.png'
)

test = pd.read_csv(os.path.join(base_path, 'test.csv'))

X_test_bearing = scaler_b.transform(test[bearing_cols].values)
X_test_nacelle = scaler_n.transform(test[nacelle_cols].values)
X_test_fused = np.hstack((X_test_nacelle, X_test_bearing))

y_test_pred_early = model_fusion.predict(X_test_fused)
y_test_labels = np.argmax(y_test_pred_early, axis=1)

submission_path = "Test_20736975.csv"
df_submission = pd.DataFrame({
    "Id": np.arange(len(y_test_labels)),
    "Predicted": y_test_labels
})

df_submission.to_csv(submission_path, index=False)