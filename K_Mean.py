import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv("OnlineRetail.csv", encoding="ISO-8859-1")

df = df.dropna()
df = df[df["Quantity"] > 0]
df = df[df["UnitPrice"] > 0]

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

customer_df = df.groupby("CustomerID").agg(
    {
        "InvoiceNo": "nunique",
        "Quantity": "sum",
        "UnitPrice": "mean"
    }
).reset_index()

X = customer_df.drop("CustomerID", axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
customer_df["Cluster"] = kmeans.fit_predict(X_scaled)

print(customer_df.head())
