import pandas as pd

# Load the CSV file
data = pd.read_csv(r"C:\Users\selin\Desktop\Correct_Data\Fused.csv")  # Replace with your actual file path

# Iterate through each column and count zeroes
columns_to_remove = []

for column in data.columns:
    num_zeroes = (data[column] == 0).sum()
    num_nonzero = (data[column] != 0).sum()

    # Mark column for removal if zeroes exceed non-zero values
    if num_zeroes > num_nonzero:
        columns_to_remove.append(column)

# Drop the marked columns
data_cleaned = data.drop(columns=columns_to_remove)

# Save the cleaned dataset
output_file = r"C:\Users\selin\Desktop\Correct_Data\Fused_Cleaned.csv"
data_cleaned.to_csv(output_file, index=False)

print(f"Removed {len(columns_to_remove)} columns with more zeroes than non-zero values.")
print(f"Cleaned dataset saved to: {output_file}")
