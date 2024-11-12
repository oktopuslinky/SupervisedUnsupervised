import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import numpy as np
from sklearn.metrics import silhouette_score

'''
def numClusters(data, max_k):
    
    #means = sum of square error
    means = []
    inertia = []
    
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters = k)
        kmeans.fit(data)

        means.append(k)
        inertia.append(kmeans.inertia_)

    #Generate elbow plot
    fig = plt.subplots(figsize=(10, 5))
    plt.plot(means, inertia)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.show()
    
    elbow_clusters = np.argmax(np.diff(means)) + 2

    silhouette_scores = []
    for k in range(2, max_k + 1):  # Silhouette requires at least 2 clusters
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data)
        silhouette_scores.append(silhouette_score(data, labels))
    
    # Plotting Silhouette Scores
    plt.figure(figsize=(8, 4))
    plt.plot(range(2, max_k + 1), silhouette_scores, marker='o')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Analysis for Optimal k")
    plt.show()

    # Finding the optimal k for Silhouette Method
    silhouette_clusters = np.argmax(silhouette_scores) + 2

    print('SC:', silhouette_clusters)
    print('EC:', elbow_clusters)

    '''

df = pd.read_csv("austin_weather2.csv")
df['DayOfYear'] = pd.to_datetime(df['Date']).dt.dayofyear

#numClusters(df[['DayOfYear', 'TempAvgF']], 10)

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
