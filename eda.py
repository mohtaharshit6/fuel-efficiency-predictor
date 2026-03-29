import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("auto-mpg-clean.csv")

# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("heatmap.png")
plt.show()

# MPG vs Horsepower
plt.figure(figsize=(6,4))
sns.scatterplot(x='horsepower', y='mpg', data=df)
plt.title("MPG vs Horsepower")
plt.savefig("mpg_vs_horsepower.png")
plt.show()

# MPG vs Weight
plt.figure(figsize=(6,4))
sns.scatterplot(x='weight', y='mpg', data=df)
plt.title("MPG vs Weight")
plt.savefig("mpg_vs_weight.png")
plt.show()

print("All plots saved! ✅")