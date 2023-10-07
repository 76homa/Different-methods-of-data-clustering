import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

# Load your dataset
df = pd.read_excel('C:\\Users\\homa.behmardi\\Downloads\\huawei.xlsx')

# Select features for clustering
X = df[['Total_Traffic(UL+DL)(GB)(Ericsson_LTE_Sector)', 'DL_Traffic(GB)(Ericsson_LTE_Sector)', 'DL_Spectral_efficiency(Ericsson_LTE_Sector)', 'Average_Reported_CQI(Ericsson_LTE_Sector)', 'DL_PRB_Utilization_Rate(Ericsson_LTE_Sector)', 'Sector_Bandwidth(MHz)(Ericsson_LTE_Sector)', 'Cell_Availability_Rate_Include_Blocking(Ericsson_LTE_Sector)', 'VoLTE_Traffic_Erlang_QCI1(Sector_EricLTE)_QCI']]

# Standardize the features (important for K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize lists to store evaluation metrics
silhouette_scores = []
davies_bouldin_scores = []
wcss = []

# Define a range of k values to try
k_values = range(2, 20)

# Perform K-Means clustering for each k value
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    
    # Calculate Silhouette Score and Davies-Bouldin Index
    silhouette = silhouette_score(X_scaled, kmeans.labels_)
    davies_bouldin = davies_bouldin_score(X_scaled, kmeans.labels_)
    
    # Append scores to lists
    silhouette_scores.append(silhouette)
    davies_bouldin_scores.append(davies_bouldin)
    
    # Calculate WCSS and append to the list
    wcss.append(kmeans.inertia_)

# Plot Silhouette Score
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(k_values, silhouette_scores, marker='o', linestyle='-')
plt.title('Silhouette Score')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')

# Plot Davies-Bouldin Index
plt.subplot(1, 3, 2)
plt.plot(k_values, davies_bouldin_scores, marker='o', linestyle='-')
plt.title('Davies-Bouldin Index')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Davies-Bouldin Index')

# Plot WCSS curve
plt.subplot(1, 3, 3)
plt.plot(k_values, wcss, marker='o', linestyle='-')
plt.title('Within-Cluster Sum of Squares (WCSS)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')

plt.tight_layout()

# Plot the "elbow point" in the WCSS curve
plt.figure(figsize=(8, 4))
plt.plot(k_values, wcss, marker='o', linestyle='-')
plt.title('WCSS Curve to Determine Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')

# Add a vertical line at the chosen k value (elbow point)
chosen_k = 4
plt.axvline(x=chosen_k, color='red', linestyle='--', label=f'Chosen k = {chosen_k}')
plt.legend()

# Perform K-Means clustering with the chosen k
final_kmeans = KMeans(n_clusters=chosen_k, random_state=42)
final_kmeans.fit(X_scaled)

# Assign cluster labels to the original data
df['Cluster_Labels'] = final_kmeans.labels_

# Create a scatter plot of clusters
plt.figure(figsize=(8, 6))
plt.scatter(df['Total_Traffic(UL+DL)(GB)(Ericsson_LTE_Sector)'], df['Average_UE_DL_Throughput(Mbps)(Ericsson_LTE_Sector)'], c=df['Cluster_Labels'], cmap='viridis')
plt.title('Clustered Data')
plt.xlabel('Total Traffic (UL+DL) GB')
plt.ylabel('Average UE DL Throughput (Mbps)')
plt.show()
