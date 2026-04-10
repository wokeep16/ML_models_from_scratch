import numpy as np
import pandas as pd
x = np.linspace(-5, 5, 1000)
y = np.sin(x) + np.random.normal(0, 0.5, 1000)

# pd.DataFrame({'x': x, 'y': y}).to_csv('data.csv', index=False)

df = pd.read_csv('data.csv')

print(df['x'].values)
