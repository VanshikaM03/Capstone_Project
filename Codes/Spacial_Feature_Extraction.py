import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model

# Load your EEG data file
# Assuming your data is in a CSV format and the last column is ground truth
eeg_data = pd.read_csv(r"C:\Users\selin\Desktop\Correct_Data\ground_truth_4.csv")  # Replace with the actual file path
eeg_data = eeg_data.iloc[:, :-1].to_numpy()  # Exclude the last column (ground truth)

# Parameters
window_size = 500  # Window size of 500 rows (samples)
n_columns = 22  # The number of features (columns in the data), excluding the ground truth column

# Extract windows of size 500x22 from the EEG data
def create_windows(data, window_size):
    windows = []
    for i in range(0, len(data) - window_size + 1, window_size):
        window = data[i:i + window_size, :]
        windows.append(window)
    return np.array(windows)

# Create the windows for spatial feature extraction
eeg_windows = create_windows(eeg_data, window_size)
print(f"Shape of EEG windows: {eeg_windows.shape}")

# CNN Model with Residual Connections for Spatial Feature Extraction
def build_cnn_residual_model(input_shape):
    inputs = layers.Input(shape=input_shape)

    # First convolutional block
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Residual block 1
    residual = x
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.add([x, residual])  # Skip connection
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Residual block 2
    residual = x
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # Project residual to match the number of filters (64)
    residual = layers.Conv2D(64, (1, 1), padding='same')(residual)

    # Add the skip connection
    x = layers.add([x, residual])
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Fully connected layer for feature extraction
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(64, activation='relu')(x)  # Output layer for spatial features

    model = Model(inputs, outputs)
    return model

# Building the CNN model
input_shape = (window_size, n_columns, 1)  # Add channel dimension for Conv2D
cnn_model = build_cnn_residual_model(input_shape)

# Compile the model
cnn_model.compile(optimizer='adam', loss='mse')

# Summarize the model
cnn_model.summary()

# Reshape EEG windows for CNN input (add the channel dimension)
eeg_windows_reshaped = eeg_windows.reshape(eeg_windows.shape[0], window_size, n_columns, 1)

# Train or use the CNN model for feature extraction
spatial_features = cnn_model.predict(eeg_windows_reshaped)
print(f"Extracted Spatial Features Shape: {spatial_features.shape}")

# Convert the extracted features to a DataFrame
spatial_features_df = pd.DataFrame(spatial_features)

# Save the spatial features to a CSV file
spatial_features_df.to_csv(r"C:\Users\selin\Desktop\Correct_Data\ground_truth_4_spacial.csv", index=False)

print("Spatial features have been saved to 'ground_truth_4_spacial.csv'.")