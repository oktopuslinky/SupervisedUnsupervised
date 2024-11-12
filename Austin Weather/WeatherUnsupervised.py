import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import numpy as np
from sklearn.metrics import silhouette_score

df = pd.read_csv("austin_weather2.csv")
df['DayOfYear'] = pd.to_datetime(df['Date']).dt.dayofyear

km = KMeans(n_clusters=4, random_state=10)
print(km)

y_predicted = km.fit_predict(df[['DayOfYear', 'TempAvgF']])
df['cluster'] = y_predicted
print(y_predicted)

df0 = df[df['cluster']==0]
df1 = df[df['cluster']==1]
df2 = df[df['cluster']==2]
df3 = df[df['cluster']==3]

plt.scatter(df0['DayOfYear'], df0['TempAvgF'], color='orange')
plt.scatter(df1['DayOfYear'], df1['TempAvgF'], color='blue')
plt.scatter(df2['DayOfYear'], df2['TempAvgF'], color='green')
plt.scatter(df3['DayOfYear'], df3['TempAvgF'], color='brown')
plt.xlabel('DayOfYear')
plt.ylabel('TempAvgF')
plt.show()
