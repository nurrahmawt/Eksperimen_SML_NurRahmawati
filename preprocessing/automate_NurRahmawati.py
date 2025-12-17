import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('houseprices_raw/house_prices.csv')

# Data Preprocessing
df.duplicated().sum()

scaler = StandardScaler()
num_cols = ["Square_Footage","Year_Built","Lot_Size"]

df[num_cols] = scaler.fit_transform(df[num_cols])
df[num_cols].head()

num_cols = df.select_dtypes(include=["int64", "float64"]).columns

for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(col, "→", outliers.shape[0], "outlier")

df.to_csv(
    "preprocessing/houses_preprocessing/house_data_processed.csv",
    index=False
)