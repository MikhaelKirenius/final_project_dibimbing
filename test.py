import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Template Notebook untuk Dashboard Penjualan
code_template = """
# --- 1. Import Library ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 2. Load Dataset ---
df = pd.read_csv("supermarket_sales.csv")

# --- 3. Preprocessing ---
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Weekday'] = df['Date'].dt.day_name()
df['Hour'] = pd.to_datetime(df['Time']).dt.hour

# --- 4. KPI Overview ---
total_revenue = df['Total'].sum()
total_gross_income = df['gross income'].sum()
num_transactions = df['Invoice ID'].nunique()
avg_rating = df['Rating'].mean()
avg_basket = df['Quantity'].mean()
aov = df['Total'].mean()

print("Total Revenue:", total_revenue)
print("Total Gross Income:", total_gross_income)
print("Jumlah Transaksi:", num_transactions)
print("Rata-rata Rating:", avg_rating)
print("Basket Size:", avg_basket)
print("AOV:", aov)

# --- 5. Analisis Waktu ---
# Revenue harian
daily_revenue = df.groupby('Date')['Total'].sum()
daily_revenue.plot(figsize=(10,4), title="Revenue Harian")
plt.show()

# Revenue per bulan
monthly_revenue = df.groupby('Month')['Total'].sum()
monthly_revenue.plot(kind='bar', title="Revenue per Bulan (Jan–Mar 2019)")
plt.show()

# Transaksi per jam
hourly_sales = df.groupby('Hour')['Invoice ID'].count()
hourly_sales.plot(kind='bar', title="Jumlah Transaksi per Jam")
plt.show()

# --- 6. Analisis Produk ---
product_sales = df.groupby('Product line')['Total'].sum().sort_values(ascending=False)
product_sales.plot(kind='bar', title="Top Product Line by Revenue")
plt.show()

# --- 7. Analisis Pelanggan ---
sns.barplot(x="Customer type", y="Total", data=df, estimator=sum)
plt.title("Revenue berdasarkan Customer Type")
plt.show()

sns.barplot(x="Gender", y="Total", data=df, estimator=sum)
plt.title("Revenue berdasarkan Gender")
plt.show()

# --- 8. Analisis Cabang ---
branch_sales = df.groupby('Branch')['Total'].sum()
branch_sales.plot(kind='bar', title="Revenue per Branch")
plt.show()

# --- 9. Analisis Pembayaran ---
payment_dist = df['Payment'].value_counts()
payment_dist.plot(kind='pie', autopct='%1.1f%%', title="Distribusi Metode Pembayaran")
plt.ylabel("")
plt.show()

# --- 10. Korelasi Numerik ---
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Korelasi antar Variabel Numerik")
plt.show()
"""

# Save template notebook as .py so user can adapt in Jupyter
template_file = "/mnt/data/dashboard_notebook_template.py"
with open(template_file, "w") as f:
    f.write(code_template)

template_file
