import pandas as pd
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

# Connect to PostgreSQL database with customer and order data
engine = create_engine("postgresql+psycopg2://localhost/revenue_db")

# Load all customer, order, and interaction data from database
customers = pd.read_sql("SELECT * FROM customers", engine)
orders = pd.read_sql("SELECT * FROM orders", engine)
interactions = pd.read_sql("SELECT * FROM interactions", engine)

# Create features that predict customer churn behavior

# Find when each customer last placed an order
last_orders = orders.groupby("customer_id")["order_date"].max().reset_index()
last_orders.columns = ["customer_id", "last_order_date"]
customers = customers.merge(last_orders, on="customer_id", how="left")

# Calculate days since last purchase (key churn indicator)
# Use 999 for customers with no orders to avoid missing data
customers["days_since_last_order"] = (
 pd.to_datetime("today") - pd.to_datetime(customers["last_order_date"])
).dt.days.fillna(999)

# Calculate average spending per order (spending pattern indicator)
order_totals = orders.groupby("customer_id")["total_amount"].mean().reset_index()
order_totals.columns = ["customer_id", "avg_order_value"]
customers = customers.merge(order_totals, on="customer_id", how="left")

# Count email engagement (engagement indicator)
email_opens = interactions[interactions["type"] == "email_open"]
interact_counts = email_opens.groupby("customer_id").size().reset_index(name="email_opens")
customers = customers.merge(interact_counts, on="customer_id", how="left")
customers["email_opens"] = customers["email_opens"].fillna(0)

# Define churn: customers inactive for more than 120 days
# Business rule: 4+ months without order = likely churned
customers["churned"] = customers["days_since_last_order"].apply(lambda x: 1 if x > 120 else 0)

# Select features that predict churn behavior
features = ["days_since_last_order", "avg_order_value", "email_opens", "total_orders", "lifetime_value"]
X = customers[features].fillna(0)  # Input features
y = customers["churned"]  # Target: churned (1) or active (0)

# Split data for training and testing model accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest model to predict churn
# 100 trees for good accuracy without overfitting
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test model performance on unseen data
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Generate churn risk scores for all customers
# Probability score (0-1): higher = more likely to churn
customers["churn_risk_score"] = model.predict_proba(X)[:, 1]

# Export customer risk scores for business action
customers[["customer_id", "churn_risk_score", "churned"] + features].to_csv("churn_scores.csv", index=False)
print("\nChurn model trained. Risk scores saved to churn_scores.csv")