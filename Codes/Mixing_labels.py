import pandas as pd
import numpy as np

# Load the temporal and spatial feature files for all four labels
# Adjust the file paths as necessary
temporal_label1 = pd.read_csv(r"C:\Users\selin\Desktop\Correct_Data\ground_truth_1_temp.csv")  # Temporal file for label 1
temporal_label2 = pd.read_csv(r"C:\Users\selin\Desktop\Correct_Data\ground_truth_2_temp.csv")  # Temporal file for label 2
temporal_label3 = pd.read_csv(r"C:\Users\selin\Desktop\Correct_Data\ground_truth_3_temp.csv")  # Temporal file for label 3
temporal_label4 = pd.read_csv(r"C:\Users\selin\Desktop\Correct_Data\ground_truth_4_temp.csv")  # Temporal file for label 4

spatial_label1 = pd.read_csv(r"C:\Users\selin\Desktop\Correct_Data\ground_truth_1_spacial.csv")  # Spatial file for label 1
spatial_label2 = pd.read_csv(r"C:\Users\selin\Desktop\Correct_Data\ground_truth_2_spacial.csv")  # Spatial file for label 2
spatial_label3 = pd.read_csv(r"C:\Users\selin\Desktop\Correct_Data\ground_truth_3_spacial.csv")  # Spatial file for label 3
spatial_label4 = pd.read_csv(r"C:\Users\selin\Desktop\Correct_Data\ground_truth_4_spacial.csv")  # Spatial file for label 4

# Concatenate temporal and spatial files for all labels
temporal_features = pd.concat(
    [temporal_label1, temporal_label2, temporal_label3, temporal_label4], ignore_index=True
)
spatial_features = pd.concat(
    [spatial_label1, spatial_label2, spatial_label3, spatial_label4], ignore_index=True
)

# Check if both datasets have the same length
if len(temporal_features) != len(spatial_features):
    raise ValueError("Temporal and Spatial feature files must have the same length.")

# Create an index array for shuffling
index_array = np.arange(len(temporal_features))

# Shuffle the index array
np.random.shuffle(index_array)

# Apply the shuffled index to both temporal and spatial features
temporal_features_shuffled = temporal_features.iloc[index_array].reset_index(drop=True)
spatial_features_shuffled = spatial_features.iloc[index_array].reset_index(drop=True)

# Save the shuffled datasets
temporal_features_shuffled.to_csv(r"C:\Users\selin\Desktop\Correct_Data\Temporal_Mixed.csv", index=False)
spatial_features_shuffled.to_csv(r"C:\Users\selin\Desktop\Correct_Data\Spacial_Mixed.csv", index=False)

print("Shuffling done. Temporal and spatial feature files with all labels have been shuffled consistently.")