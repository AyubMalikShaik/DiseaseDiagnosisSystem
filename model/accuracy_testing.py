import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# Define the chunk size
chunk_size = 50000  # Number of rows to read

# Filepath for the test dataset
test_file_path = "../Dataset/dataset_comb.csv"  # Replace with your test dataset path

# Load the models
random_forest_model = joblib.load("./rand_forest_model.pkl")
logistic_regression_model = joblib.load("./logistic_regression_model.pkl")
ridge_like_model = joblib.load("./ridge_like_model.pkl")  # SGDClassifier trained incrementally

# Load pretrained Naive Bayes and SVM models
naive_bayes_model = joblib.load("./naive_bayes_model.pkl")
svm_model = joblib.load("./svm_like_model.pkl")

# Load the trained Neural Network models
nn_model = tf.keras.models.load_model("./neural_net_model.h5")  # Neural network model

# Read the first chunk of the test dataset
chunk = pd.read_csv(test_file_path, chunksize=chunk_size)
first_chunk = next(chunk)  # Get the first chunk

# Split into features and labels
X_test_chunk = first_chunk.iloc[:, 1:]  # Features
y_test_chunk = first_chunk.iloc[:, 0]   # Labels

# Ensure that the input features are numeric (convert if necessary)

# Evaluate models
rf_pred = random_forest_model.predict(X_test_chunk)
lr_pred = logistic_regression_model.predict(X_test_chunk)
ridge_pred = ridge_like_model.predict(X_test_chunk)
nb_pred = naive_bayes_model.predict(X_test_chunk)
svm_pred = svm_model.predict(X_test_chunk)

# Neural network predictions
nn_pred = (nn_model.predict(X_test_chunk) > 0.5).astype(int)  # Assuming binary classification, threshold 0.5
label_encoder = LabelEncoder()
b = label_encoder.fit_transform(y_test_chunk.values.ravel()) 
# Calculate accuracies
rf_accuracy = accuracy_score(y_test_chunk, rf_pred)
lr_accuracy = accuracy_score(y_test_chunk, lr_pred)
ridge_accuracy = accuracy_score(y_test_chunk, ridge_pred)
nb_accuracy = accuracy_score(y_test_chunk, nb_pred)
svm_accuracy = accuracy_score(y_test_chunk, svm_pred)
nn_accuracy = accuracy_score(y_test_chunk, nn_pred)

# Print results
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")
print(f"Ridge-like Classifier Accuracy: {ridge_accuracy:.4f}")
print(f"Naive Bayes Accuracy: {nb_accuracy:.4f}")
print(f"SVM Accuracy: {svm_accuracy:.4f}")
print(f"Neural Network Accuracy: {nn_accuracy:.4f}")
