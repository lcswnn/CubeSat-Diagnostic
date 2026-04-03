import pandas as pd

df = pd.read_excel('data/telemetry.xlsx')
df.to_csv('telemetry.csv', index=False)
print("Done!")