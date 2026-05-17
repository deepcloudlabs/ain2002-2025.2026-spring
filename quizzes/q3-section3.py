import json
import pandas as pd

orders_json = """[
    {"OrderID": 101, "Customer": "Ali", "City": "Istanbul", "Amount": 2500, "Status": "Completed"},
    {"OrderID": 102, "Customer": "Ayse", "City": "Ankara", "Amount": null, "Status": "Completed"},
    {"OrderID": 103, "Customer": "Mehmet", "City": "Izmir", "Amount": 1200, "Status": "Cancelled"},
    {"OrderID": 104, "Customer": "Zeynep", "City": "Istanbul", "Amount": 3100, "Status": "Completed"},
    {"OrderID": 105, "Customer": "Can", "City": "Bursa", "Amount": null, "Status": "Pending"},
    {"OrderID": 106, "Customer": "Elif", "City": "Ankara", "Amount": 1800, "Status": "Completed"}
]"""

orders_data = json.loads(orders_json)

df = pd.DataFrame(orders_data)

print(df.head(5))

print(df.isnull().sum())

amount_median = df['Amount'].median()
df['Amount'] = df['Amount'].fillna(amount_median)

df['High ValueOrder'] = df['Amount'] >= 2000

completed_orders = df[df['Status'] == 'Completed']
print(completed_orders)

city_avg_amount = df.groupby('City')['Amount'].mean()
print(city_avg_amount)

df.to_json('cleaned_orders.json', orient='records', indent=4)
print("\nPipeline complete: Final DataFrame written to 'cleaned_orders.json'.")