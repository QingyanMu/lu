import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({'Height': ['170', '180', '155', '160', '175', '177', '190', '140'],
'Weight':['60', '88.5', '55', '65', '80', '95', '78.4', '45']})


# Convert Height column to int64 and Weight to float64
df['Height'] = pd.to_numeric(df['Height'])

df['Weight'] = pd.to_numeric(df['Weight'])

print(df.dtypes)


# Plot columns
df[['Height', 'Weight']].plot()

plt.show()
