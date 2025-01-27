import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
customers = pd.read_csv("C:\Users\hansi\OneDrive\Desktop\zeotap_assessment\ecommerce_analysis\data\Customers.csv")
transactions = pd.read_csv("C:\Users\hansi\OneDrive\Desktop\zeotap_assessment\ecommerce_analysis\data\Transactions.csv")

# Aggregate transaction data
transaction_summary = transactions.groupby('CustomerID')['TotalValue'].sum().reset_index()

# Merge with customer data
customer_data = pd.merge(customers, transaction_summary, on='CustomerID', how='left').fillna(0)

# Apply clustering
kmeans = KMeans(n_clusters=3, random_state=42)
customer_data['Cluster'] = kmeans.fit_predict(customer_data[['TotalValue']])

# Calculate DB Index
db_index = davies_bouldin_score(customer_data[['TotalValue']], customer_data['Cluster'])
print(f"Davies-Bouldin Index: {db_index}")

# Visualize clusters
sns.scatterplot(data=customer_data, x='TotalValue', y='Cluster', hue='Cluster', palette='viridis')
plt.title("Customer Clusters")
plt.savefig("../outputs/customer_clusters.png")
print("Clustering complete. Outputs saved.")
