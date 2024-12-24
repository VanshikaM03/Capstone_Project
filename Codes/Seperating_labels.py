import pandas as pd
import os

# Specify the directory where you want to save the split files
save_directory = r'C:\Users\selin\Desktop\Correct_Data'

# Read the rawdata.csv file into a DataFrame
df = pd.read_csv(r"C:\Users\selin\Desktop\Correct_Data\Normalizednew.csv")

# Ensure that the 'GT' column is treated as a numeric type to avoid string misinterpretation
gt_column = df.columns[-1]
df[gt_column] = pd.to_numeric(df[gt_column], errors='coerce')  # This will convert non-numeric values to NaN

# Split the DataFrame based on the GT column into 4 files
for label in [1, 2, 3, 4]:
    # Filter the DataFrame based on the current label
    label_df = df[df[gt_column] == label]

    # Define the file path for saving the split data
    file_path = os.path.join(save_directory, f'label_{label}.csv')

    # Save the filtered DataFrame to a new CSV file in the specified directory
    label_df.to_csv(file_path, index=False)

    print(f"Saved {file_path} with {len(label_df)} rows.")

print(f"Files have been split and saved to {save_directory}.")
