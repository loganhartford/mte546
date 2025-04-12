import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

inhouse = pd.read_csv("regression_data/inhouse_sensor21.csv")
client = pd.read_csv("regression_data/client_sensor21.csv")

def desired_output(q1, q2):
    return -0.0081 * q1 + 0.0589 * q2 - 0.000625

def estimated_output(q1, q2):
    return 0.017 * q1 + 0.042 * q2 + 0.0015

inhouse['desired'] = desired_output(inhouse['q1'], inhouse['q2'])
inhouse['estimated'] = estimated_output(inhouse['q1'], inhouse['q2'])

client['desired'] = desired_output(client['q1'], client['q2'])
client['estimated'] = estimated_output(client['q1'], client['q2'])

plt.figure(figsize=(10, 5))
plt.plot(inhouse['time'], inhouse['desired'], label='Desired Output')
plt.plot(inhouse['time'], inhouse['estimated'], label='Estimated Output')
plt.xlabel('Time (s)')
plt.ylabel('Sensor Output')
plt.title('Desired vs Estimated Output - Inhouse Sensor 21')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('inhouse_timeseries.png')

plt.figure(figsize=(6, 6))
plt.scatter(inhouse['desired'], inhouse['estimated'], alpha=0.5, label='Inhouse Sensor 21')
plt.scatter(client['desired'], client['estimated'], alpha=0.5, label='Client Sensor 21')
plt.plot([-0.04, 0.04], [-0.04, 0.04], 'k--', label='Perfect Model Line')
plt.xlabel('Desired Output')
plt.ylabel('Estimated Output')
plt.title('Desired vs Estimated Scatterplot')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.savefig('scatterplot_outputs.png')