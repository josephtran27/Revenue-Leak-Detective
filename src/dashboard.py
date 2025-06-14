import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load ML model results from previous scripts
churn_df = pd.read_csv("churn_scores.csv")  # Customer churn risk scores
anomalies_df = pd.read_csv("weekly_anomalies.csv")  # Weekly revenue anomalies

# Prepare anomaly data for filtering by year
anomalies_df["week_start"] = pd.to_datetime(anomalies_df["week_start"])
anomalies_df["year"] = anomalies_df["week_start"].dt.year

# Configure dashboard layout and title
st.set_page_config(page_title="Revenue Leak Detective", layout="wide")
st.title("Revenue Leak Detective")
st.markdown("Analyze churn risk and detect weekly revenue anomalies using ML models.")

# Get unique values for filter dropdowns (safely handle missing columns)
tiers = churn_df["tier"].unique().tolist() if "tier" in churn_df.columns else []
countries = churn_df["country"].unique().tolist() if "country" in churn_df.columns else []

# Create sidebar filters for churn analysis
st.sidebar.header("Churn Risk Filters")
selected_tiers = st.sidebar.multiselect("Select Tier(s):", tiers, default=tiers) if tiers else []
selected_countries = st.sidebar.multiselect("Select Country(s):", countries, default=countries) if countries else []

# Slider for lifetime value range filtering
lv_range = st.sidebar.slider(
"Lifetime Value Range:",
float(churn_df["lifetime_value"].min()),
float(churn_df["lifetime_value"].max()),
 (float(churn_df["lifetime_value"].min()), float(churn_df["lifetime_value"].max()))
)

# Create sidebar filters for anomaly analysis
st.sidebar.header("Anomaly Filters")
years = sorted(anomalies_df["year"].unique())
selected_year = st.sidebar.selectbox("Select Year:", years)
show_only_anomalies = st.sidebar.checkbox("Show Only Anomalies", value=False)

# Display key business metrics at top of dashboard
col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", f"{len(churn_df):,}")
col2.metric("Avg Churn Risk", f"{churn_df['churn_risk_score'].mean():.2f}")
col3.metric("Anomalous Weeks", f"{anomalies_df['is_anomaly'].sum()}")

# Create two main tabs for different analysis views
tab1, tab2 = st.tabs(["Churn Risk Table", "Weekly Revenue Anomalies"])

# Tab 1: Customer churn risk analysis
with tab1:
 st.subheader("Churn Risk Explorer")
 
 # Apply user-selected filters to churn data
 filtered_churn = churn_df.copy()
 if tiers:
     filtered_churn = filtered_churn[filtered_churn["tier"].isin(selected_tiers)]
 if countries:
     filtered_churn = filtered_churn[filtered_churn["country"].isin(selected_countries)]
 
 # Filter by lifetime value range
 filtered_churn = filtered_churn[
     filtered_churn["lifetime_value"].between(lv_range[0], lv_range[1])
 ]
 
 # Sort by highest churn risk first (most urgent customers)
 filtered_churn = filtered_churn.sort_values("churn_risk_score", ascending=False)
 
 # Display filtered results in interactive table
 st.dataframe(filtered_churn, use_container_width=True)
 
 # Allow users to download filtered data for further analysis
 csv = filtered_churn.to_csv(index=False).encode("utf-8")
 st.download_button("Download Churn Risk CSV", csv, "churn_scores.csv", "text/csv")

# Tab 2: Revenue anomaly analysis
with tab2:
 st.subheader(f"Revenue Over Time ({selected_year})")
 
 # Filter anomalies by selected year
 filtered_anomalies = anomalies_df[anomalies_df["year"] == selected_year]
 
 # Optionally show only anomalous weeks
 if show_only_anomalies:
     filtered_anomalies = filtered_anomalies[filtered_anomalies["is_anomaly"] == 1]
 
 # Create revenue trend chart with anomaly highlights
 fig, ax = plt.subplots(figsize=(12, 5))
 ax.plot(filtered_anomalies["week_start"], filtered_anomalies["total_amount"], label="Weekly Revenue")
 
 # Highlight anomalous weeks with red dots
 flagged = filtered_anomalies[filtered_anomalies["is_anomaly"] == 1]
 ax.scatter(flagged["week_start"], flagged["total_amount"], color="red", label="Anomaly")
 
 # Format chart for business presentation
 ax.set_title("Weekly Revenue (Red = Anomaly)")
 ax.set_xlabel("Week")
 ax.set_ylabel("Total Revenue")
 ax.legend()
 plt.xticks(rotation=45)  # Rotate dates for readability
 
 # Display chart in dashboard
 st.pyplot(fig)
 
 # Allow users to download anomaly data
 csv2 = filtered_anomalies.to_csv(index=False).encode("utf-8")
 st.download_button("Download Anomaly CSV", csv2, "weekly_anomalies.csv", "text/csv")