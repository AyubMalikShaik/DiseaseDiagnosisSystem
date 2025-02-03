import pandas as pd

# Load your CSV file
df = pd.read_csv("./Dataset/dataset_comb.csv")  # Replace with your file name

# Group by 'Disease' and apply a logical OR (max) for each symptom column
consolidated_df = df.groupby('diseases').max()

# Reset index if needed
consolidated_df.reset_index(inplace=True)

# Save to a new CSV file (optional)
consolidated_df.to_csv("consolidated_disease_symptoms.csv", index=False)

print("Consolidation complete! Here's the resulting DataFrame:")
print(consolidated_df.head())
