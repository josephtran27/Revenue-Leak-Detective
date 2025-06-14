import pandas as pd
from sqlalchemy import create_engine
from sklearn.ensemble import IsolationForest
from datetime import datetime
import matplotlib.pyplot as plt

# Connect to PostgreSQL database containing revenue data
engine = create_engine("postgresql+psycopg2://localhost/revenue_db")

# Pull all completed orders with dates and amounts
# Only 'completed' status ensures we're analyzing actual revenue
query = """
 SELECT order_date, total_amount
 FROM orders
 WHERE status = 'completed'
"""
df = pd.read_sql(query, engine)

# Convert text dates to proper datetime format for calculations
df["order_date"] = pd.to_datetime(df["order_date"])

# Group orders by week starting on Monday
# dayofweek: Monday=0, Sunday=6, so we subtract to get week start
df["week_start"] = df["order_date"] - pd.to_timedelta(df["order_date"].dt.dayofweek, unit="d")

# Sum up total revenue for each week
weekly_revenue = df.groupby("week_start")["total_amount"].sum().reset_index()

# Use machine learning to find unusual revenue weeks
# contamination=0.1 means expect 10% of weeks to be anomalies
model = IsolationForest(contamination=0.1, random_state=42)

# Get anomaly scores: -1 = anomaly, 1 = normal
weekly_revenue["anomaly_score"] = model.fit_predict(weekly_revenue[["total_amount"]])

# Convert scores to binary flag: 1 = anomaly, 0 = normal
weekly_revenue["is_anomaly"] = weekly_revenue["anomaly_score"].apply(lambda x: 1 if x == -1 else 0)

# Export results for business team analysis
weekly_revenue.to_csv("weekly_anomalies.csv", index=False)
print("Weekly revenue anomalies saved to weekly_anomalies.csv")

# Create visual chart showing revenue trends with anomalies highlighted
plt.figure(figsize=(12, 6))

# Plot weekly revenue as a line chart
plt.plot(weekly_revenue["week_start"], weekly_revenue["total_amount"], label="Weekly Revenue")

# Highlight anomalous weeks with red dots
plt.scatter(
 weekly_revenue[weekly_revenue["is_anomaly"] == 1]["week_start"],
 weekly_revenue[weekly_revenue["is_anomaly"] == 1]["total_amount"],
 color="red",
 label="Anomaly",
)

# Add chart labels and formatting
plt.title("Weekly Revenue with Anomaly Detection")
plt.xlabel("Week Start")
plt.ylabel("Total Revenue")
plt.xticks(rotation=45)  # Rotate dates for readability
plt.legend()
plt.tight_layout()  # Prevent label cutoff

# Save chart as image file for reports
plt.savefig("weekly_anomalies_plot.png")
print("Anomaly plot saved as weekly_anomalies_plot.png")