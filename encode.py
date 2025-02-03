from sklearn.preprocessing import LabelEncoder
import pandas as pd
df=pd.read_csv('./Dataset/dataset_comb.csv')
# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Apply LabelEncoder to the 'diseases' column
df['diseases'] = label_encoder.fit_transform(df['diseases'])

# Save the updated dataset
output_path = './Dataset/encoded_dataset.csv'
df.to_csv(output_path, index=False)

# Display the first few rows of the updated dataset
df.head(), output_path
