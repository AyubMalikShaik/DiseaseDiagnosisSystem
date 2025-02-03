import pandas as pd

# Specify the CSV file path and the output Parquet file path
csv_file_path = './Dataset/dataset_comb.csv'
parquet_file_path = './Dataset/parquet_file.parquet'

# Read the CSV file in chunks if it's very large
chunk_size = 10**6  # Adjust this based on your memory capacity

# Use a loop to process the file in chunks and write it to Parquet
first_chunk = True
for chunk in pd.read_csv(csv_file_path, chunksize=chunk_size):
    if first_chunk:
        chunk.to_parquet(parquet_file_path, engine='pyarrow', index=False)
        first_chunk = False
    else:
        chunk.to_parquet(parquet_file_path, engine='pyarrow', index=False, append=True)

print(f"CSV file has been converted to Parquet at {parquet_file_path}")
