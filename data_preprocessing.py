import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Load the data
print("Loading data...")
df = pd.read_csv('all_stocks_5yr.csv')

# Basic data cleaning
print("\nCleaning data...")
# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])
# Remove any rows with missing values
df = df.dropna()

# Display original data info
print("\nOriginal Data Info:")
print(f"Number of rows: {len(df)}")
print(f"Number of columns: {len(df.columns)}")
print("\nOriginal Data Sample:")
print(df.head())

# 1. Convert stock names to binary
print("\nConverting stock names to binary...")
# Get unique stock names
unique_stocks = df['Name'].unique()
# Create a dictionary mapping stock names to numbers
stock_to_number = {stock: i for i, stock in enumerate(unique_stocks)}
# Create new column with binary encoding
df['Name_binary'] = df['Name'].map(stock_to_number)

# Display binary encoding mapping
print("\nBinary Encoding Mapping (first 10 stocks):")
for i, (stock, number) in enumerate(stock_to_number.items()):
    if i < 10:  # Show only first 10 mappings
        print(f"{stock}: {number}")

# 2. Scale the numerical features
print("\nScaling numerical features...")
# Select numerical columns
numeric_columns = ['open', 'high', 'low', 'close', 'volume']

# Create a copy of the original values before scaling
df_original = df[numeric_columns].copy()

# Scale the numerical columns
scaler = MinMaxScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Display scaling results
print("\nScaling Results:")
print("Original vs Scaled values (first 5 rows):")
comparison_df = pd.DataFrame({
    'Original_Open': df_original['open'].head(),
    'Scaled_Open': df['open'].head(),
    'Original_Close': df_original['close'].head(),
    'Scaled_Close': df['close'].head()
})
print(comparison_df)

# Display summary statistics
print("\nSummary Statistics of Scaled Data:")
print(df[numeric_columns].describe())

# Save preprocessed data
print("\nSaving preprocessed data...")
df.to_csv('preprocessed_data.csv', index=False)
print("Preprocessed data saved to 'preprocessed_data.csv'")

# Save the stock name mapping
print("\nSaving stock name mapping...")
mapping_df = pd.DataFrame({
    'Stock_Name': list(stock_to_number.keys()),
    'Binary_Code': list(stock_to_number.values())
})
mapping_df.to_csv('stock_name_mapping.csv', index=False)
print("Stock name mapping saved to 'stock_name_mapping.csv'")

# Display final data structure
print("\nFinal Data Structure:")
print("Columns in the preprocessed dataset:")
for col in df.columns:
    print(f"- {col}")

# Display sample of final preprocessed data
print("\nFinal Preprocessed Data Sample (first 5 rows):")
print(df.head())

# Verify the preprocessing
print("\nVerification:")
print(f"Number of unique stocks: {len(stock_to_number)}")
print(f"Range of binary codes: {min(stock_to_number.values())} to {max(stock_to_number.values())}")
print(f"Range of scaled values: {df[numeric_columns].min().min():.2f} to {df[numeric_columns].max().max():.2f}") 