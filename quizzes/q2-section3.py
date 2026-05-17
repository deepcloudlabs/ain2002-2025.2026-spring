import pandas as pd

sales_data = {
    'Region': ['North', 'South', 'North', 'East', 'South', 'East', 'North', 'South'],
    'Q1': [120, 85, 200, 95, 60, 175, 110, 90],
    'Q2': [135, 90, 190, 80, 75, 160, 130, 95],
    'Q3': [150, 70, 210, 100, 85, 180, 125, 80]
}

sales = pd.DataFrame(sales_data)
print(sales)

print(sales[['Q1', 'Q2', 'Q3']].describe())

max_q1_index = sales['Q1'].idxmax()
min_q1_index = sales['Q1'].idxmin()
print(f"\nRow Index with Highest Q1 Sales (idxmax): {max_q1_index}")
print(f"Row Index with Lowest Q1 Sales (idxmin): {min_q1_index}")

sales['Q1_cumulative'] = sales['Q1'].cumsum()
print(sales)

print(sales[['Q1', 'Q2', 'Q3']].corr())

q1_q3_covariance = sales['Q1'].cov(sales['Q3'])
print(f"\nSample Covariance Between Q1 and Q3: {q1_q3_covariance}")

distinct_regions = sales['Region'].unique()
region_frequencies = sales['Region'].value_counts()
filtered_sales_mask = sales['Region'].isin(['North', 'East'])
filtered_sales_df = sales[filtered_sales_mask]

print(distinct_regions)
print(region_frequencies)
print(filtered_sales_df)