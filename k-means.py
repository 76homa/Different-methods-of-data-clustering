import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

# Load your dataset
df = pd.read_excel('C:\\Users\\homa.behmardi\\Downloads\\huawei.xlsx')

# Select features for clustering
X = df[['Total_Traffic(UL+DL)(GB)(Ericsson_LTE_Sector)', 'DL_Traffic(GB)(Ericsson_LTE_Sector)', 'DL_Spectral_efficiency(Ericsson_LTE_Sector)', 'Average_Reported_CQI(Ericsson_LTE_Sector)', 'DL_PRB_Utilization_Rate(Ericsson_LTE_Sector)', 'Sector_Bandwidth(MHz)(Ericsson_LTE_Sector)', 'Cell_Availability_Rate_Include_Blocking(Ericsson_LTE_Sector)']]  # Replace with your feature columns

# Standardize the features (important for K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize lists to store evaluation metrics
silhouette_scores = []
davies_bouldin_scores = []

# Define a range of k values to try (number of clusters)
k_values = range(2, 6)  # You can adjust the range as needed

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
    
    # Print evaluation metrics for each k
    print(f'For k={k}:')
    print(f'Silhouette Score: {silhouette:.4f}')
    print(f'Davies-Bouldin Index: {davies_bouldin:.4f}\n')

# Based on the plots and scores, choose the optimal number of clusters (k)
# For example, look for the "elbow" point in the Silhouette Score or Davies-Bouldin Index curve

# After selecting the optimal k value, you can perform K-Means clustering as follows:

# Perform K-Means clustering with the chosen k value
chosen_k = 4  # Replace with your chosen k value
kmeans = KMeans(n_clusters=chosen_k, random_state=42)
kmeans.fit(X_scaled)

# Assign cluster labels to the original data
df['Cluster_Labels'] = kmeans.labels_

# Visualize the clustered data using a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df['Total_Traffic(UL+DL)(GB)(Ericsson_LTE_Sector)'], df['Average_UE_DL_Throughput(Mbps)(Ericsson_LTE_Sector)'], c=df['Cluster_Labels'], cmap='viridis')
plt.xlabel('Total Traffic (UL+DL) (GB)')
plt.ylabel('Average UE DL Throughput (Mbps)')
plt.title('K-Means Clustering Results')
plt.colorbar(label='Cluster')
plt.show()
