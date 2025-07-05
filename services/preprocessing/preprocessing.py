import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical



# Set paths locally
import os


# Use raw strings (r"") to avoid backslash issues
RAW_DATA_PATH = r"C:\Users\phili\epilepsy_mlops\data\raw\Epileptic Seizure Recognition.csv"
PROCESSED_X_TRAIN_PATH = r"C:\Users\phili\epilepsy_mlops\data\processed\X_train.npy"
PROCESSED_Y_TRAIN_PATH = r"C:\Users\phili\epilepsy_mlops\data\processed\Y_train.npy"
PROCESSED_X_TEST_PATH = r"C:\Users\phili\epilepsy_mlops\data\processed\X_test.npy"
PROCESSED_Y_TEST_PATH = r"C:\Users\phili\epilepsy_mlops\data\processed\Y_test.npy"
PATIENT_DATA_PATH = r"C:\Users\phili\epilepsy_mlops\data\patients\patients_data.csv"



# Create directories if they don't exist
os.makedirs(os.path.dirname(PROCESSED_X_TRAIN_PATH), exist_ok=True)
os.makedirs(os.path.dirname(PROCESSED_Y_TRAIN_PATH), exist_ok=True)
os.makedirs(os.path.dirname(PROCESSED_X_TEST_PATH), exist_ok=True)
os.makedirs(os.path.dirname(PROCESSED_Y_TEST_PATH), exist_ok=True)
os.makedirs(os.path.dirname(PATIENT_DATA_PATH), exist_ok=True)

print(f"📂 Reading raw data from: {RAW_DATA_PATH}")

# Read the raw dataset
try:
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"✅ Data loaded with shape: {df.shape}")
except Exception as e:
    print(f"❌ Error loading data: {str(e)}")
    raise

# Remove the first column which is typically an index
df = df.iloc[:, 1:]

# Map labels to binary: 1 stays 1, 2-5 become 0
df.iloc[:, -1] = df.iloc[:, -1].apply(lambda x: 1 if x == 1 else 0)

# Separate the data into classes based on the modified labels (1 or 0)
df_label_1 = df[df.iloc[:, -1] == 1]  # label 1 (epilepsy)
df_label_0 = df[df.iloc[:, -1] == 0]  # labels 2-5 (no epilepsy)

print(f"📊 Class distribution: Epilepsy: {len(df_label_1)}, Non-epilepsy: {len(df_label_0)}")

# Sample 400 random entries from each class
df_label_1_sample = df_label_1.sample(n=400, random_state=42)
df_label_0_sample = df_label_0.sample(n=400, random_state=42)

# Combine the two sampled dataframes
df_train = pd.concat([df_label_1_sample, df_label_0_sample])

# Shuffle the combined dataset to ensure randomness
df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

# Separate features (X) and labels (y)
X_train_full = df_train.iloc[:, :-1].values  # All columns except the last one
y_train_full = df_train.iloc[:, -1].values   # Last column is the label

# One-hot encode the labels
Y_train_full = to_categorical(y_train_full, num_classes=2)

# Split into train/test sets (80% train, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(
    X_train_full, Y_train_full, test_size=0.20, random_state=42
)

# Reshape for LSTM model (adding a dimension for compatibility)
X_train = X_train.reshape(-1, 178, 1)
X_test = X_test.reshape(-1, 178, 1)

# Save the processed data
print(f"💾 Saving processed data to {os.path.dirname(PROCESSED_X_TRAIN_PATH)}")
np.save(PROCESSED_X_TRAIN_PATH, X_train)
np.save(PROCESSED_Y_TRAIN_PATH, Y_train)
np.save(PROCESSED_X_TEST_PATH, X_test)
np.save(PROCESSED_Y_TEST_PATH, Y_test)

# Save the remaining data to a CSV file (for patient data)
df_remaining = df.drop(df_train.index)
df_remaining.to_csv(PATIENT_DATA_PATH, index=False)

# Print the shapes for verification
print(f"✅ Preprocessing completed!")
print(f"📊 X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
print(f"📊 X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")
print(f"📊 Patient dataset saved at {PATIENT_DATA_PATH}")
print(f"Class distribution in training data: {pd.Series(y_train_full).value_counts()}")
