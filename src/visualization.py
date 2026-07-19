import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/train.csv")

# Train-wise dataset
train_df = df.groupby("Train_No").agg({
    "Distance": "max",
    "SN": "count"
}).reset_index()

train_df.rename(columns={"SN": "Stops"}, inplace=True)

# Distance vs Stops
plt.figure(figsize=(8,5))
plt.scatter(train_df["Distance"], train_df["Stops"])
plt.xlabel("Distance (km)")
plt.ylabel("Number of Stops")
plt.title("Distance vs Number of Stops")
plt.grid(True)
plt.show()
plt.figure(figsize=(8,5))
plt.hist(train_df["Distance"], bins=20)
plt.xlabel("Distance")
plt.ylabel("Frequency")
plt.title("Distribution of Distance")
plt.show()