# Import libraries
import pandas as pd
from matplotlib import pyplot as plt



# Function
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


# Load Data
df = pd.read_csv("iyzico_data.csv")

# Overview
check_df(df)

# What are the start and end dates of the dataset
df["transaction_date"].min()
# Timestamp('2018-01-01 00:00:00')
df["transaction_date"].max()

# Timestamp('2020-12-31 00:00:00')


# We calculated the total number of transactions at each merchant
df["merchant_id"].unique()

# We calculated the total amount of payment in each merchant
df.groupby("merchant_id").agg({"Total_Paid": "sum"})

# To visualize the transaction count of member merchants in each year
for id in df.merchant_id.unique():
    plt.figure(figsize=(15, 15))
    plt.subplot(3, 1, 1, title=str(id) + ' 2018-2019 Transaction Count')
    df[(df.merchant_id == id) & (df.transaction_date >= "2018-01-01") & (df.transaction_date < "2019-01-01")][
        "Total_Transaction"].plot()
    plt.xlabel('')
    plt.subplot(3, 1, 2, title=str(id) + ' 2019-2020 Transaction Count')
    df[(df.merchant_id == id) & (df.transaction_date >= "2019-01-01") & (df.transaction_date < "2020-01-01")][
        "Total_Transaction"].plot()
    plt.xlabel('')
    plt.show(block=True)
