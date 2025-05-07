import pandas as pd

df = pd.read_csv('metric_results.csv')

metrics_only = df.drop(columns=['sample'])

mean_values = metrics_only.mean()

print("Mean values:")
print(mean_values)
