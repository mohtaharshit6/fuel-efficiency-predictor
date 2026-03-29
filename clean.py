import pandas as pd

df = pd.read_csv("auto-mpg.csv")

df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')

df = df.dropna()

df = df.drop(columns=['car name'])

print("Cleaned Shape:", df.shape)
print("\nMissing Values After Cleaning:\n", df.isnull().sum())
print("\nData Types:\n", df.dtypes)

df.to_csv("auto-mpg-clean.csv", index=False)
print("\nCleaned file saved as auto-mpg-clean.csv ✅")