import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Set the working directory
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of this script
os.chdir(script_dir)  # Change the working directory to the script's location

# Load the dataset
file_path = "wholesale_customers.csv"  # Name of the dataset file
if os.path.exists(file_path):
    print("File found successfully!")
else:
    print("File not found! Please check the file path.")
    exit()

data = pd.read_csv(file_path)

# Inspect the dataset
print("First 5 rows of the dataset:")
print(data.head())
print("\nDataset Information:")
print(data.info())

# Select features (exclude the first two columns, assuming they are non-numerical identifiers)
features = data.iloc[:, 2:]  # First two columns are excluded as they are likely identifiers or categories

# Preprocessing: Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Determine the optimal number of clusters using the elbow method
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

# Visualize the elbow method
plt.figure(figsize=(8, 4))
plt.plot(k_range, inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method to Determine Optimal k')
plt.show()

# Apply K-means with the optimal number of clusters
optimal_k = 3  # Choose the optimal k from the elbow plot
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(scaled_features)

# Add cluster labels to the dataset
data['Cluster'] = clusters

# Calculate Silhouette Score
silhouette_avg = silhouette_score(scaled_features, clusters)
print(f"Silhouette Score: {silhouette_avg:.2f}")

# Analyze the clusters
print("\nAverage values for each cluster:")
cluster_means = data.groupby('Cluster').mean()
print(cluster_means)

# Visualize the clusters using 2D PCA
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_features)
data['PCA1'] = pca_components[:, 0]
data['PCA2'] = pca_components[:, 1]

plt.figure(figsize=(8, 6))
for cluster in range(optimal_k):
    cluster_data = data[data['Cluster'] == cluster]
    plt.scatter(cluster_data['PCA1'], cluster_data['PCA2'], label=f"Cluster {cluster}")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.title("K-means Clustering Results (2D PCA)")
plt.legend()
plt.show()
