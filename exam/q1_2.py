import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

inhouse = pd.read_csv("regression_data/inhouse_sensor21.csv")
client = pd.read_csv("regression_data/client_sensor21.csv")

def compute_desired(q1, q2):
    return -0.0081 * q1 + 0.0589 * q2 - 0.000625

inhouse['desired'] = compute_desired(inhouse['q1'], inhouse['q2'])
client['desired'] = compute_desired(client['q1'], client['q2'])

combined = pd.concat([inhouse, client], ignore_index=True)

X = combined[['v1', 'v2']].values
y = combined['desired'].values

model = LinearRegression()
model.fit(X, y)

coef_q1, coef_q2 = model.coef_
intercept = model.intercept_

y_pred = model.predict(X)
residuals = y - y_pred
residual_std = np.std(residuals)

print(f"Model equation: ŷ = {coef_q1:.6f}*v1 + {coef_q2:.6f}*v2 + {intercept:.6f} + ε")
print(f"Estimated noise standard deviation: {residual_std:.6f}")
print(f"Residuals mean: {np.mean(residuals):.6f}")

plt.hist(residuals, bins=50, alpha=0.7)
plt.title("Residuals of Linear Model")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig("residual_histogram.png")
plt.show()
