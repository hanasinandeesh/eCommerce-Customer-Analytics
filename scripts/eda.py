# ecommerce_analysis/scripts/eda.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
customers = pd.read_csv("../data/Customers.csv")
products = pd.read_csv("../data/Products.csv")
transactions = pd.read_csv("../data/Transactions.csv")

# Data Inspection
print(customers.info())
print(products.info())
print(transactions.info())

# EDA
def perform_eda():
    # Summary stats
    print(customers.describe())
    print(products.describe())
    print(transactions.describe())
    
    # Visualizations
    sns.countplot(data=customers, x='Region')
    plt.title("Customer Count by Region")
    plt.savefig("../outputs/customer_region_distribution.png")
    
    sns.boxplot(data=products, x='Category', y='Price')
    plt.title("Product Price by Category")
    plt.savefig("../outputs/product_price_category.png")
    
    print("EDA complete. Outputs saved to /outputs folder.")

perform_eda()
