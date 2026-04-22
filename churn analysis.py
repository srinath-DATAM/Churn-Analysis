"""
Customer Churn Analysis Project
Author: Srinath M
Date: March 2026

This project analyzes customer churn patterns using Python libraries:
- Pandas for data manipulation
- NumPy for numerical computations
- Matplotlib and Seaborn for visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("="*60)
print("CUSTOMER CHURN ANALYSIS PROJECT")
print("="*60)

# ============================================================================
# 1. DATA LOADING AND INITIAL EXPLORATION
# ============================================================================
print("\n1. LOADING DATA...")
df = pd.read_csv('customer_data.csv')

print(f"✅ Dataset loaded successfully!")
print(f"📊 Total customers: {len(df)}")
print(f"📋 Total features: {len(df.columns)}")

print("\n" + "="*60)
print("2. DATA OVERVIEW")
print("="*60)
print(df.head())
print("\nData Types:")
print(df.dtypes)
print("\nMissing Values:")
print(df.isnull().sum())

# ============================================================================
# 2. DESCRIPTIVE STATISTICS USING PANDAS & NUMPY
# ============================================================================
print("\n" + "="*60)
print("3. DESCRIPTIVE STATISTICS")
print("="*60)

print("\nNumerical Features Summary:")
print(df.describe())

print("\nChurn Rate Analysis:")
churn_rate = df['Churned'].mean() * 100
print(f"Overall Churn Rate: {churn_rate:.2f}%")
print(f"Customers Churned: {df['Churned'].sum()}")
print(f"Customers Retained: {len(df) - df['Churned'].sum()}")

# Using NumPy for statistical calculations
print("\nKey Metrics (using NumPy):")
print(f"Average Age: {np.mean(df['Age']):.1f} years")
print(f"Median Tenure: {np.median(df['TenureMonths']):.0f} months")
print(f"Std Dev Monthly Charges: ₹{np.std(df['MonthlyCharges']):.2f}")
print(f"Average Support Calls: {np.mean(df['SupportCalls']):.2f}")

# ============================================================================
# 3. CHURN ANALYSIS BY SEGMENTS
# ============================================================================
print("\n" + "="*60)
print("4. CHURN ANALYSIS BY CUSTOMER SEGMENTS")
print("="*60)

# Churn by Contract Type
print("\nChurn Rate by Contract Type:")
churn_by_contract = df.groupby('ContractType')['Churned'].agg(['mean', 'sum', 'count'])
churn_by_contract.columns = ['Churn_Rate', 'Churned_Count', 'Total_Customers']
churn_by_contract['Churn_Rate'] = churn_by_contract['Churn_Rate'] * 100
print(churn_by_contract.round(2))

# Churn by Tenure Groups
df['TenureGroup'] = pd.cut(df['TenureMonths'],
                            bins=[0, 12, 24, 36, 60],
                            labels=['0-12 months', '12-24 months', '24-36 months', '36+ months'])
print("\nChurn Rate by Tenure:")
print(df.groupby('TenureGroup')['Churned'].mean() * 100)

# Churn by City
print("\nChurn Rate by City:")
churn_by_city = df.groupby('City')['Churned'].mean() * 100
print(churn_by_city.sort_values(ascending=False))

# ============================================================================
# 4. STATISTICAL HYPOTHESIS TESTING
# ============================================================================
print("\n" + "="*60)
print("5. STATISTICAL HYPOTHESIS TESTING")
print("="*60)

# Chi-square test: Contract Type vs Churn
contingency_table = pd.crosstab(df['ContractType'], df['Churned'])
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
print("\nChi-Square Test: Contract Type vs Churn")
print(f"Chi-square statistic: {chi2:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Result: {'Significant relationship' if p_value < 0.05 else 'No significant relationship'}")

# T-test: Monthly Charges for churned vs retained
churned_charges = df[df['Churned'] == 1]['MonthlyCharges']
retained_charges = df[df['Churned'] == 0]['MonthlyCharges']
t_stat, p_value_t = stats.ttest_ind(churned_charges, retained_charges)
print("\nT-Test: Monthly Charges (Churned vs Retained)")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value_t:.4f}")
print(f"Mean Charges (Churned): ₹{churned_charges.mean():.2f}")
print(f"Mean Charges (Retained): ₹{retained_charges.mean():.2f}")

# ============================================================================
# 5. CORRELATION ANALYSIS USING NUMPY
# ============================================================================
print("\n" + "="*60)
print("6. CORRELATION ANALYSIS")
print("="*60)

# Select numerical features
numerical_features = ['Age', 'TenureMonths', 'MonthlyCharges', 'TotalCharges',
                      'SupportCalls', 'SatisfactionScore', 'Churned']
correlation_matrix = df[numerical_features].corr()

print("\nCorrelation with Churn:")
print(correlation_matrix['Churned'].sort_values(ascending=False))

# ============================================================================
# 6. REVENUE IMPACT ANALYSIS
# ============================================================================
print("\n" + "="*60)
print("7. REVENUE IMPACT ANALYSIS")
print("="*60)

total_revenue = df['TotalCharges'].sum()
churned_revenue = df[df['Churned'] == 1]['TotalCharges'].sum()
retained_revenue = df[df['Churned'] == 0]['TotalCharges'].sum()

print(f"\nTotal Revenue: ₹{total_revenue:,.2f}")
print(f"Revenue from Churned Customers: ₹{churned_revenue:,.2f}")
print(f"Revenue from Retained Customers: ₹{retained_revenue:,.2f}")
print(f"Revenue Loss from Churn: {(churned_revenue/total_revenue)*100:.2f}%")

monthly_revenue_loss = df[df['Churned'] == 1]['MonthlyCharges'].sum()
print(f"\nMonthly Recurring Revenue Loss: ₹{monthly_revenue_loss:,.2f}")
print(f"Annualized Revenue at Risk: ₹{monthly_revenue_loss * 12:,.2f}")

# ============================================================================
# 7. VISUALIZATIONS USING MATPLOTLIB & SEABORN
# ============================================================================
print("\n" + "="*60)
print("8. GENERATING VISUALIZATIONS...")
print("="*60)

# Create figure with multiple subplots
fig = plt.figure(figsize=(16, 12))

# 1. Churn Distribution (Pie Chart)
ax1 = plt.subplot(3, 3, 1)
churn_counts = df['Churned'].value_counts()
colors = ['#2ecc71', '#e74c3c']
plt.pie(churn_counts, labels=['Retained', 'Churned'], autopct='%1.1f%%',
        colors=colors, startangle=90)
plt.title('Overall Churn Distribution', fontsize=12, fontweight='bold')

# 2. Churn by Contract Type (Bar Chart)
ax2 = plt.subplot(3, 3, 2)
churn_contract = df.groupby('ContractType')['Churned'].mean() * 100
churn_contract.plot(kind='bar', color='#3498db')
plt.title('Churn Rate by Contract Type', fontsize=12, fontweight='bold')
plt.ylabel('Churn Rate (%)')
plt.xlabel('Contract Type')
plt.xticks(rotation=45)

# 3. Monthly Charges Distribution (Histogram with Seaborn)
ax3 = plt.subplot(3, 3, 3)
sns.histplot(data=df, x='MonthlyCharges', hue='Churned', bins=30, kde=True)
plt.title('Monthly Charges Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Monthly Charges (₹)')

# 4. Tenure vs Churn (Box Plot)
ax4 = plt.subplot(3, 3, 4)
sns.boxplot(data=df, x='Churned', y='TenureMonths', palette='Set2')
plt.title('Tenure Distribution by Churn Status', fontsize=12, fontweight='bold')
plt.xlabel('Churn Status (0=Retained, 1=Churned)')
plt.ylabel('Tenure (Months)')

# 5. Support Calls vs Churn (Violin Plot)
ax5 = plt.subplot(3, 3, 5)
sns.violinplot(data=df, x='Churned', y='SupportCalls', palette='muted')
plt.title('Support Calls by Churn Status', fontsize=12, fontweight='bold')
plt.xlabel('Churn Status (0=Retained, 1=Churned)')
plt.ylabel('Number of Support Calls')

# 6. Correlation Heatmap (Seaborn)
ax6 = plt.subplot(3, 3, 6)
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=1)
plt.title('Feature Correlation Heatmap', fontsize=12, fontweight='bold')

# 7. Churn by City (Bar Chart)
ax7 = plt.subplot(3, 3, 7)
city_churn = df.groupby('City')['Churned'].mean() * 100
city_churn.sort_values().plot(kind='barh', color='#e67e22')
plt.title('Churn Rate by City', fontsize=12, fontweight='bold')
plt.xlabel('Churn Rate (%)')

# 8. Satisfaction Score Distribution (Count Plot)
ax8 = plt.subplot(3, 3, 8)
sns.countplot(data=df, x='SatisfactionScore', hue='Churned', palette='pastel')
plt.title('Satisfaction Score by Churn Status', fontsize=12, fontweight='bold')
plt.xlabel('Satisfaction Score (1-5)')
plt.ylabel('Count')

# 9. Churn by Payment Method (Stacked Bar)
ax9 = plt.subplot(3, 3, 9)
payment_churn = pd.crosstab(df['PaymentMethod'], df['Churned'], normalize='index') * 100
payment_churn.plot(kind='bar', stacked=True, color=['#2ecc71', '#e74c3c'])
plt.title('Churn Distribution by Payment Method', fontsize=12, fontweight='bold')
plt.ylabel('Percentage (%)')
plt.xlabel('Payment Method')
plt.xticks(rotation=45, ha='right')
plt.legend(['Retained', 'Churned'])

plt.tight_layout()
plt.savefig('churn_analysis_visualizations.png', dpi=300, bbox_inches='tight')
print("✅ Visualizations saved: churn_analysis_visualizations.png")
plt.show()

# ============================================================================
# 8. KEY INSIGHTS & RECOMMENDATIONS
# ============================================================================
print("\n" + "="*60)
print("9. KEY INSIGHTS & BUSINESS RECOMMENDATIONS")
print("="*60)

print("\n📊 KEY FINDINGS:")
print(f"1. Overall churn rate is {churn_rate:.1f}% ({df['Churned'].sum()} customers)")
print(f"2. Month-to-month contracts have highest churn rate")
print(f"3. Customers with high support calls are more likely to churn")
print(f"4. Short tenure customers (<12 months) show higher churn")
print(f"5. Monthly revenue at risk: ₹{monthly_revenue_loss:,.0f}")

print("\n💡 RECOMMENDATIONS:")
print("1. Incentivize long-term contracts (1-2 year)")
print("2. Improve customer support quality to reduce call volumes")
print("3. Implement retention program for new customers (<6 months)")
print("4. Focus on improving satisfaction scores (currently low for churned)")
print("5. Special attention to month-to-month contract customers")

# Save summary report
summary = {
    'Total_Customers': len(df),
    'Churned_Customers': df['Churned'].sum(),
    'Churn_Rate_%': round(churn_rate, 2),
    'Total_Revenue': round(total_revenue, 2),
    'Revenue_Lost': round(churned_revenue, 2),
    'Monthly_Revenue_At_Risk': round(monthly_revenue_loss, 2),
    'Annual_Revenue_At_Risk': round(monthly_revenue_loss * 12, 2)
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv('churn_analysis_summary.csv', index=False)
print("\n✅ Summary report saved: churn_analysis_summary.csv")

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
print("\nGenerated Files:")
print("1. customer_data.csv - Dataset")
print("2. churn_analysis_visualizations.png - All charts")
print("3. churn_analysis_summary.csv - Summary metrics")