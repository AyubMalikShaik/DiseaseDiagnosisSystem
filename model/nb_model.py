import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import shuffle
import joblib

# Load the dataset
df_comb = pd.read_csv("../Dataset/dataset_comb.csv")

# Split features and labels
a = df_comb.iloc[:, 1:]  # Features
b = df_comb.iloc[:, 0]   # Labels

# Initialize MultinomialNB model
nb_model = MultinomialNB()

# Split dataset into chunks
chunk_size = 10000  # Process in chunks of 10,000 samples
num_samples = a.shape[0]
num_chunks = (num_samples + chunk_size - 1) // chunk_size  # Calculate total chunks

# Get unique classes (required for `partial_fit`)
classes = b.unique()

# Shuffle the dataset
a, b = shuffle(a, b, random_state=42)

# Train the Naive Bayes model in chunks
for i in range(num_chunks):
    start_idx = i * chunk_size
    end_idx = min(start_idx + chunk_size, num_samples)
    
    # Get the chunk
    chunk_a = a.iloc[start_idx:end_idx]
    chunk_b = b[start_idx:end_idx]
    
    # Fit the model incrementally
    nb_model.partial_fit(chunk_a, chunk_b, classes=classes)
    
    print(f"Completed training chunk {i + 1} of {num_chunks}")

# Save the trained model
joblib.dump(nb_model, "./naive_bayes_model.pkl")
print("Naive Bayes model saved successfully!")
