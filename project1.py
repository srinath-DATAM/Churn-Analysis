import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Number of customers
n_customers = 1000

# Generate customer data
print("Generating customer dataset...")

# Customer IDs
customer_ids = [f'CUST{str(i).zfill(4)}' for i in range(1, n_customers + 1)]

# Demographics
ages = np.random.randint(18, 70, n_customers)
genders = np.random.choice(['Male', 'Female'], n_customers, p=[0.52, 0.48])
cities = np.random.choice(['Chennai', 'Bangalore', 'Mumbai', 'Delhi', 'Hyderabad', 'Pune'], n_customers, p=[0.25, 0.20, 0.18, 0.15, 0.12, 0.10])

# Account information
tenure_months = np.random.randint(1, 60, n_customers)
monthly_charges = np.round(np.random.uniform(500, 5000, n_customers), 2)

# Service usage
internet_service = np.random.choice(['Fiber', 'DSL', 'No'], n_customers, p=[0.45, 0.40, 0.15])
online_security = np.random.choice(['Yes', 'No'], n_customers, p=[0.35, 0.65])
tech_support = np.random.choice(['Yes', 'No'], n_customers, p=[0.30, 0.70])
streaming_tv = np.random.choice(['Yes', 'No'], n_customers, p=[0.40, 0.60])
contract_type = np.random.choice(['Month-to-month', 'One year', 'Two year'], n_customers, p=[0.55, 0.25, 0.20])
payment_method = np.random.choice(['Electronic check', 'Credit card', 'Bank transfer', 'Mailed check'], n_customers, p=[0.35, 0.30, 0.25, 0.10])

# Customer support
support_calls = np.random.poisson(2, n_customers)
complaints = np.random.binomial(1, 0.25, n_customers)

# Generate churn (realistic patterns)
churn_probability = np.zeros(n_customers)

# Higher churn for:
# - Short tenure
churn_probability += (tenure_months < 6) * 0.3
# - Month-to-month contracts
churn_probability += (contract_type == 'Month-to-month') * 0.25
# - High support calls
churn_probability += (support_calls > 3) * 0.2
# - Complaints
churn_probability += complaints * 0.25
# - No online security
churn_probability += (online_security == 'No') * 0.1
# - High monthly charges
churn_probability += (monthly_charges > 4000) * 0.15

# Add some randomness
churn_probability += np.random.uniform(0, 0.1, n_customers)

# Cap at 1.0
churn_probability = np.minimum(churn_probability, 1.0)

# Generate actual churn
churned = (np.random.random(n_customers) < churn_probability).astype(int)

# Total charges based on tenure and monthly charges
total_charges = np.round(monthly_charges * tenure_months * np.random.uniform(0.95, 1.05, n_customers), 2)

# Customer satisfaction score (1-5)
satisfaction_score = np.where(churned == 1,
                              np.random.choice([1, 2, 3], n_customers, p=[0.4, 0.4, 0.2]),
                              np.random.choice([3, 4, 5], n_customers, p=[0.3, 0.4, 0.3]))

# Create DataFrame
df = pd.DataFrame({
    'CustomerID': customer_ids,
    'Age': ages,
    'Gender': genders,
    'City': cities,
    'TenureMonths': tenure_months,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'ContractType': contract_type,
    'PaymentMethod': payment_method,
    'SupportCalls': support_calls,
    'Complaints': complaints,
    'SatisfactionScore': satisfaction_score,
    'Churned': churned
})

# Save to CSV
df.to_csv('customer_data.csv', index=False)

print(f"✅ Dataset created: {n_customers} customers")
print(f"✅ Churn rate: {churned.mean()*100:.1f}%")
print(f"✅ File saved: customer_data.csv")
print("\nDataset preview:")
print(df.head(10))
print("\nDataset info:")
print(df.info())
print("\nChurn distribution:")
print(df['Churned'].value_counts())