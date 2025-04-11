import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def compute_desired(q1, q2):
    return -0.0081 * q1 + 0.0589 * q2 - 0.000625


def original_model(v1, v2):
    return 0.017 * v1 + 0.042 * v2 + 0.0015


def two_a_model(v1, v2):
    return 0.010957 * v1 + 0.029573 * v2 -0.000235


client22 = pd.read_csv("regression_data/client_sensor22.csv")
client22['desired'] = compute_desired(client22['q1'], client22['q2'])


client22['orig_pred'] = original_model(client22['v1'], client22['v2'])
client22['two_a_pred'] = two_a_model(client22['v1'], client22['v2'])


def compute_metrics(true, pred):
    error = true - pred
    return {
        "Average Error": np.mean(error),
        "Variance of Error": np.var(error),
        "RMSE": np.sqrt(mean_squared_error(true, pred)),
        "R2": r2_score(true, pred)
    }


orig_metrics = compute_metrics(client22['desired'], client22['orig_pred'])
two_a_metrics = compute_metrics(client22['desired'], client22['two_a_pred'])


print("\nOriginal Model Metrics:")
for k, v in orig_metrics.items():
    print(f"{k}: {v:.6f}")

print("\n2A Model Metrics:")
for k, v in two_a_metrics.items():
    print(f"{k}: {v:.6f}")


client23 = pd.read_csv("regression_data/client_sensor23.csv")
pred_23 = two_a_model(client23['v1'], client23['v2'])
pred_23.to_csv("outputs_client_sensor_23.csv", index=False, header=False)

print("\nPredictions for client_sensor_23.csv saved to outputs_client_sensor_23.csv")
