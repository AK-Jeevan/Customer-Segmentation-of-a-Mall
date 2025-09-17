# Dividing different types of Customers into groups of a Mall using K-Means Clustering
# Evaluating the optimal number of clusters using Silhouette Score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns

data = pd.read_csv('Mall_Customers.csv')
print(data.head())
print(data.describe())
data.dropna(inplace=True)

# Drop non-numeric columns for clustering
x = data.drop(['Gender'], axis=1)

# Feature scaling
s = StandardScaler()
x_scaled = s.fit_transform(x)

# Find optimal number of clusters using the elbow method and silhouette score
wss = []
silhouette_scores = []
K = range(2, 11)  # silhouette score is not defined for k=1

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(x_scaled)
    wss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(x_scaled, labels))

# Plot WSS (Elbow Method)
plt.figure(figsize=(8, 4))
plt.plot(K, wss, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Within-Cluster Sum of Squares (WSS)')
plt.grid(True)
plt.show()

# Plot Silhouette Scores
plt.figure(figsize=(8, 4))
plt.plot(K, silhouette_scores, marker='s', color='orange')
plt.title('Silhouette Score For Optimal k')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()

# Choose optimal k (for example, the one with highest silhouette score)
optimal_k = K[np.argmax(silhouette_scores)]
print(f"Optimal number of clusters based on silhouette score: {optimal_k}")

# Fit final KMeans
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(x_scaled)

# Add cluster labels to the original data
data['Cluster'] = clusters

# Visualize clusters using the first two principal components
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=x_pca[:, 0], y=x_pca[:, 1], hue=clusters, palette='Set1', s=100)
plt.title('Customer Segments (PCA-reduced)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend(title='Cluster')
plt.show()

# Show a few customers from each cluster
for i in range(optimal_k):
    print(f"\nCluster {i} customers:")
    print(data[data['Cluster'] == i].head())