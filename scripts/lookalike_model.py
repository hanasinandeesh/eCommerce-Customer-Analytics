# ecommerce_analysis/scripts/lookalike_model.py
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Load data
customers = pd.read_csv("../data/Customers.csv")
transactions = pd.read_csv("../data/Transactions.csv")

# Aggregate transaction data
transaction_summary = transactions.groupby('CustomerID')['TotalValue'].sum().reset_index()

# Merge with customer data
customer_data = pd.merge(customers, transaction_summary, on='CustomerID', how='left').fillna(0)

# Scale data for similarity calculation
scaler = StandardScaler()
scaled_data = scaler.fit_transform(customer_data[['TotalValue']])

# Compute similarity
similarity_matrix = cosine_similarity(scaled_data)
similarity_df = pd.DataFrame(similarity_matrix, index=customer_data['CustomerID'], columns=customer_data['CustomerID'])

# Get top 3 lookalikes
lookalikes = {}
for customer in customer_data['CustomerID'][:20]:
    similar_customers = similarity_df[customer].sort_values(ascending=False)[1:4]
    lookalikes[customer] = list(similar_customers.items())

# Save results
lookalikes_df = pd.DataFrame.from_dict(lookalikes, orient='index', columns=['Lookalike1', 'Lookalike2', 'Lookalike3'])
lookalikes_df.to_csv("../outputs/FirstName_LastName_Lookalike.csv")
print("Lookalike model complete. Results saved.")
