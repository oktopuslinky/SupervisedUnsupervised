import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Cancer_Data.csv")
print(df)

# target
y = df['diagnosis']
# features without NaN and target
X = df.drop(columns=['diagnosis', 'id']).dropna(axis=1)


#SCALE DATA
scaler = StandardScaler()
scaler.fit(X)
scaled_df = scaler.transform(X)
print(scaled_df)



#PCA
pca = PCA(n_components=2)
pca.fit(scaled_df)
x_pca = pca.transform(scaled_df)
print(scaled_df.shape)
print(x_pca.shape)

#PLOT
color_map = {'M': 'red', 'B': 'green'}
colors = y.map(color_map)
plt.figure(figsize=(10,10))
plt.scatter(x_pca[:,0], x_pca[:,1], c=colors)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

#IDENTIFY RELEVANT FEATURES:

features = X.columns

#loadings are PCA components
pc1_loadings = pca.components_[0]
pc2_loadings = pca.components_[1]

#sorted dataFrame of feature loadings
loading_df_sorted_pc1 = pd.DataFrame({
    'Feature': features,
    'PC1 Loading': pc1_loadings
    }).sort_values(by='PC1 Loading', ascending=False)

loading_df_sorted_pc2 = pd.DataFrame({
    'Feature': features,
    'PC2 Loading': pc2_loadings
    }).sort_values(by='PC2 Loading', ascending=False)

print("Main features for PC1:\n", loading_df_sorted_pc1)
print("\n")
print("Main features for PC2:\n", loading_df_sorted_pc2)
