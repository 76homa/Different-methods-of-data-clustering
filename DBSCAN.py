#first: comment part 2 and run part2
#then comment part 1 and use epsilon choosen

#######################################################        PART1        ###################################################
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Load your dataset
df = pd.read_excel('C:\\Users\\homa.behmardi\\Downloads\\huawei.xlsx')

# Select features for clustering
X = df[['Total_Traffic(UL+DL)(GB)(Ericsson_LTE_Sector)', 'DL_Traffic(GB)(Ericsson_LTE_Sector)', 'DL_Spectral_efficiency(Ericsson_LTE_Sector)', 'Average_Reported_CQI(Ericsson_LTE_Sector)', 'DL_PRB_Utilization_Rate(Ericsson_LTE_Sector)', 'Sector_Bandwidth(MHz)(Ericsson_LTE_Sector)', 'Cell_Availability_Rate_Include_Blocking(Ericsson_LTE_Sector)']]  # Replace with your feature columns

# Standardize the features (important for DBSCAN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize lists to store evaluation metrics
silhouette_scores = []
davies_bouldin_scores = []

# Define a range of epsilon values to try (DBSCAN parameter)
epsilon_values = [0.2, 0.5, 0.8]  # You can adjust the range as needed

# Perform DBSCAN clustering for each epsilon value
for epsilon in epsilon_values:
    dbscan = DBSCAN(eps=epsilon)
    dbscan.fit(X_scaled)
    
    # Calculate Silhouette Score and Davies-Bouldin Index (if applicable)
    try:
        silhouette = silhouette_score(X_scaled, dbscan.labels_)
    except ValueError:
        silhouette = None
    davies_bouldin = davies_bouldin_score(X_scaled, dbscan.labels_)
    
    # Append scores to lists
    silhouette_scores.append(silhouette)
    davies_bouldin_scores.append(davies_bouldin)
    
    # Print evaluation metrics for each epsilon
    print(f'For epsilon={epsilon}:')
    if silhouette is not None:
        print(f'Silhouette Score: {silhouette:.4f}')
    print(f'Davies-Bouldin Index: {davies_bouldin:.4f}\n')

# Plot Silhouette Score (if applicable)
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epsilon_values, silhouette_scores, marker='o', linestyle='-')
plt.title('Silhouette Score')
plt.xlabel('Epsilon')
plt.ylabel('Silhouette Score')

# Plot Davies-Bouldin Index
plt.subplot(1, 2, 2)
plt.plot(epsilon_values, davies_bouldin_scores, marker='o', linestyle='-')
plt.title('Davies-Bouldin Index')
plt.xlabel('Epsilon')
plt.ylabel('Davies-Bouldin Index')

plt.tight_layout()
plt.show()

# Based on the plots and scores, choose the optimal epsilon value
# For example, look for a high silhouette score or a low Davies-Bouldin Index

# After selecting the optimal epsilon value, you can perform DBSCAN clustering as follows:
    
#########################################################        PART2         ###################################################
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Load your dataset
df = pd.read_excel('C:\\Users\\homa.behmardi\\Downloads\\huawei.xlsx')

# Select features for clustering
X = df[['Total_Traffic(UL+DL)(GB)(Ericsson_LTE_Sector)', 'DL_Traffic(GB)(Ericsson_LTE_Sector)', 'DL_Spectral_efficiency(Ericsson_LTE_Sector)', 'Average_Reported_CQI(Ericsson_LTE_Sector)', 'DL_PRB_Utilization_Rate(Ericsson_LTE_Sector)', 'Sector_Bandwidth(MHz)(Ericsson_LTE_Sector)', 'Cell_Availability_Rate_Include_Blocking(Ericsson_LTE_Sector)']]  # Replace with your feature columns

# Standardize the features (important for DBSCAN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform DBSCAN clustering with the chosen epsilon value
chosen_epsilon = 0.2  # Replace with your chosen epsilon value
dbscan = DBSCAN(eps=chosen_epsilon)
dbscan.fit(X_scaled)

# Assign cluster labels to the original data
df['Cluster_Labels'] = dbscan.labels_

# Visualize the clustered data using a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df['Total_Traffic(UL+DL)(GB)(Ericsson_LTE_Sector)'], df['Average_UE_DL_Throughput(Mbps)(Ericsson_LTE_Sector)'], c=df['Cluster_Labels'], cmap='viridis')
plt.xlabel('Total Traffic (UL+DL) (GB)')
plt.ylabel('Average UE DL Throughput (Mbps)')
plt.title('DBSCAN Clustering Results')
plt.colorbar(label='Cluster')
plt.show()

