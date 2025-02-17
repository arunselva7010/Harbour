import pandas as pd

# Load the CSV file
file_path = '/mnt/data/properties.csv'
df = pd.read_csv(file_path)

# Select the first 20 rows
df = df.head(20)

# Remove columns from K to AL (K is the 11th column, AL is the 38th column)
df.drop(df.iloc[:, 10:38], axis=1, inplace=True)

# Save the modified DataFrame to a new CSV file
output_path = '/mnt/data/properties_modified.csv'
df.to_csv(output_path, index=False)

print("The modified CSV file has been saved as 'properties_modified.csv'.")
