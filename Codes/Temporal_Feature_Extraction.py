import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Input, Dense, Add
from tensorflow.keras.models import Model

# Load the data
file_path = r"C:\Users\selin\Desktop\Correct_Data\ground_truth_4.csv"  # Adjust file path for your data
data = pd.read_csv(file_path)

# Separate the EEG data and ground truth
eeg_data = data.iloc[:, :-1].values  # first 22 columns
ground_truth = data.iloc[:, -1].values  # last column

# Function to create blocks of 500 rows
def create_blocks(data, block_size=500):
    num_blocks = data.shape[0] // block_size
    blocks = []
    for i in range(num_blocks):
        block = data[i * block_size:(i + 1) * block_size]
        blocks.append(block)
    return np.array(blocks)

# Create blocks of 500 rows
blocks = create_blocks(eeg_data, block_size=500)
# Reshape for LSTM input
blocks = blocks.reshape(blocks.shape[0], blocks.shape[1], blocks.shape[2])

# Build LSTM with skip connections for feature extraction
input_layer = Input(shape=(500, 22))  # Update input shape to match 22 channels
lstm_out = LSTM(64, return_sequences=False)(input_layer)  # Output shape: (64,)
dense_out = Dense(64)(lstm_out)  # Dense layer to match LSTM output shape

# Skip connection
skip_connection = Add()([lstm_out, dense_out])

# Final output layer (optional dimensionality reduction)
output_layer = Dense(32)(skip_connection)

# Define the model
model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='mse')

# Extract features
extracted_features = model.predict(blocks)

# Convert features into a DataFrame
features_df = pd.DataFrame(extracted_features)

# Save the extracted features to a new CSV file
features_df.to_csv(r"C:\Users\selin\Desktop\Correct_Data\ground_truth_4_temp.csv", index=False)

print("Feature extraction completed. Features saved to 'ground_truth_4_temp.csv'.")