import pandas as pd
import numpy as np
from tensorflow.keras import layers, Model, Input

# Load Temporal and Spatial Feature files, excluding the ground truth column
temporal_data = pd.read_csv(r"C:\Users\selin\Desktop\Correct_Data\Temporal_Mixed.csv")
spatial_data = pd.read_csv(r"C:\Users\selin\Desktop\Correct_Data\Spacial_Cleaned.csv")

temporal_features = temporal_data.iloc[:, :-1].to_numpy()  # Exclude ground truth (last column)
spatial_features = spatial_data.iloc[:, :-1].to_numpy()  # Exclude ground truth (last column)

# Ensure the same number of samples
assert temporal_features.shape[0] == spatial_features.shape[0], "Temporal and spatial features must have the same number of samples"

# Reshape features to match for CNN processing if needed
n_samples = temporal_features.shape[0]

# Define the input shapes for the temporal and spatial features
temporal_input = Input(shape=(temporal_features.shape[1], 1), name="Temporal_Input")
spatial_input = Input(shape=(spatial_features.shape[1], 1), name="Spatial_Input")

# Pad the temporal features to match the spatial feature dimensions (if necessary)
if temporal_features.shape[1] < spatial_features.shape[1]:
    padding = spatial_features.shape[1] - temporal_features.shape[1]
    padded_temporal = layers.ZeroPadding1D(padding=(0, padding))(temporal_input)
else:
    padded_temporal = temporal_input

# Temporal Feature Pyramid
temporal_scale1 = layers.Conv1D(64, 3, activation='relu', padding='same')(padded_temporal)
temporal_scale2 = layers.Conv1D(64, 5, activation='relu', padding='same')(padded_temporal)
temporal_scale3 = layers.Conv1D(64, 7, activation='relu', padding='same')(padded_temporal)

# Spatial Feature Pyramid
spatial_scale1 = layers.Conv1D(64, 3, activation='relu', padding='same')(spatial_input)
spatial_scale2 = layers.Conv1D(64, 5, activation='relu', padding='same')(spatial_input)
spatial_scale3 = layers.Conv1D(64, 7, activation='relu', padding='same')(spatial_input)

# Global Average Pooling to reduce dimensions
temporal_scale1 = layers.GlobalAveragePooling1D()(temporal_scale1)
temporal_scale2 = layers.GlobalAveragePooling1D()(temporal_scale2)
temporal_scale3 = layers.GlobalAveragePooling1D()(temporal_scale3)

spatial_scale1 = layers.GlobalAveragePooling1D()(spatial_scale1)
spatial_scale2 = layers.GlobalAveragePooling1D()(spatial_scale2)
spatial_scale3 = layers.GlobalAveragePooling1D()(spatial_scale3)

# Feature Fusion using Pyramid Fusion (Concatenate features from different scales)
merged_scale1 = layers.concatenate([temporal_scale1, spatial_scale1], axis=-1)
merged_scale2 = layers.concatenate([temporal_scale2, spatial_scale2], axis=-1)
merged_scale3 = layers.concatenate([temporal_scale3, spatial_scale3], axis=-1)

# Final merged representation
fused_features = layers.concatenate([merged_scale1, merged_scale2, merged_scale3], axis=-1)

# Add dense layers for refinement
dense_layer = layers.Dense(256, activation='relu')(fused_features)
dense_layer = layers.Dropout(0.3)(dense_layer)  # Dropout for regularization
dense_layer = layers.BatchNormalization()(dense_layer)
dense_output = layers.Dense(128, activation='relu')(dense_layer)

# Define the model
fusion_model = Model(inputs=[temporal_input, spatial_input], outputs=dense_output)

# Compile the model
fusion_model.compile(optimizer='adam', loss='mse')

# Print the model summary
fusion_model.summary()

# Reshape the input features to match the input shape of the model
temporal_features_reshaped = temporal_features.reshape(n_samples, temporal_features.shape[1], 1)
spatial_features_reshaped = spatial_features.reshape(n_samples, spatial_features.shape[1], 1)

# Get the fused features by passing the data through the model
fused_features_output = fusion_model.predict([temporal_features_reshaped, spatial_features_reshaped])

# Convert the fused features to a DataFrame and save to CSV
fused_features_df = pd.DataFrame(fused_features_output)
fused_features_df.to_csv(r"C:\Users\selin\Desktop\Correct_Data\Fused.csv", index=False)

print("Fused features have been saved to 'Fused.csv'.")