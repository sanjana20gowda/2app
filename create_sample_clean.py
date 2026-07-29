import pandas as pd

# Read the cleaned dataset
df = pd.read_csv("data\Clean_APL_Logistics.csv")

# Take 1000 random rows
sample_df = df.sample(n=1000, random_state=42)

# Save the sample
sample_df.to_csv("sample_Clean_APL_Logistics.csv", index=False)

print("Sample Clean dataset created successfully!")