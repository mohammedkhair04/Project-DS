import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Set style for better visualizations
plt.style.use('seaborn')
sns.set_palette("husl")

# Load the data
print("Loading data...")
df = pd.read_csv('all_stocks_5yr.csv')
df['date'] = pd.to_datetime(df['date'])

# Data Preprocessing
print("\nPreprocessing data...")
# 1. Convert stock names to numbers using LabelEncoder
le = LabelEncoder()
df['Name_numeric'] = le.fit_transform(df['Name'])

# 2. Scale the numerical features
scaler = MinMaxScaler()
numeric_columns = ['open', 'high', 'low', 'close', 'volume']
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Print preprocessing information
print("\nPreprocessing Summary:")
print(f"Number of unique stocks: {len(le.classes_)}")
print("\nStock Name to Number Mapping (first 5):")
for i, name in enumerate(le.classes_[:5]):
    print(f"{name}: {i}")

print("\nScaling Summary:")
print("All numerical features (open, high, low, close, volume) have been scaled to range [0,1]")

# Create a new DataFrame with preprocessed data
preprocessed_df = df.copy()

# Calculate additional features
print("\nCalculating additional features...")
preprocessed_df['daily_return'] = preprocessed_df.groupby('Name')['close'].pct_change() * 100
preprocessed_df['volatility'] = preprocessed_df.groupby('Name')['daily_return'].rolling(window=20).std().reset_index(0, drop=True)

# Visualizations
print("\nCreating visualizations...")

# 1. Correlation Matrix
plt.figure(figsize=(12, 8))
correlation_matrix = preprocessed_df[numeric_columns + ['daily_return']].corr()
sns.heatmap(correlation_matrix, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            fmt='.2f')
plt.title('Correlation Matrix of Scaled Features')
plt.tight_layout()
plt.show()

# 2. Stock Name Distribution
plt.figure(figsize=(15, 6))
sns.countplot(data=preprocessed_df, x='Name_numeric')
plt.title('Distribution of Stock Names (Numeric Encoding)')
plt.xlabel('Numeric Stock Code')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Price Distribution by Stock
plt.figure(figsize=(15, 6))
sns.boxplot(data=preprocessed_df, x='Name_numeric', y='close')
plt.title('Scaled Price Distribution by Stock')
plt.xlabel('Numeric Stock Code')
plt.ylabel('Scaled Price')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4. Volume vs Price Scatter
plt.figure(figsize=(12, 8))
plt.scatter(preprocessed_df['volume'], 
           preprocessed_df['close'], 
           alpha=0.5, 
           c=preprocessed_df['daily_return'], 
           cmap='viridis')
plt.colorbar(label='Daily Return (%)')
plt.title('Scaled Volume vs Price (colored by Daily Return)')
plt.xlabel('Scaled Volume')
plt.ylabel('Scaled Price')
plt.grid(True)
plt.tight_layout()
plt.show()

# 5. Volatility Distribution
plt.figure(figsize=(12, 6))
sns.histplot(data=preprocessed_df, x='volatility', bins=50)
plt.title('Distribution of 20-Day Rolling Volatility')
plt.xlabel('Volatility')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 6. Daily Returns Distribution
plt.figure(figsize=(12, 6))
sns.histplot(data=preprocessed_df, x='daily_return', bins=50)
plt.title('Distribution of Daily Returns')
plt.xlabel('Daily Return (%)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 7. Price Trends for Top 5 Stocks
top_stocks = preprocessed_df['Name'].value_counts().head(5).index
plt.figure(figsize=(15, 8))
for stock in top_stocks:
    stock_data = preprocessed_df[preprocessed_df['Name'] == stock]
    plt.plot(stock_data['date'], stock_data['close'], label=stock)
plt.title('Scaled Price Trends for Top 5 Stocks')
plt.xlabel('Date')
plt.ylabel('Scaled Price')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 8. Volume Trends
plt.figure(figsize=(15, 6))
for stock in top_stocks:
    stock_data = preprocessed_df[preprocessed_df['Name'] == stock]
    plt.plot(stock_data['date'], stock_data['volume'], label=stock, alpha=0.7)
plt.title('Scaled Volume Trends for Top 5 Stocks')
plt.xlabel('Date')
plt.ylabel('Scaled Volume')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 9. Price vs Volatility
plt.figure(figsize=(12, 8))
plt.scatter(preprocessed_df['close'], 
           preprocessed_df['volatility'], 
           alpha=0.5, 
           c=preprocessed_df['daily_return'], 
           cmap='viridis')
plt.colorbar(label='Daily Return (%)')
plt.title('Price vs Volatility (colored by Daily Return)')
plt.xlabel('Scaled Price')
plt.ylabel('Volatility')
plt.grid(True)
plt.tight_layout()
plt.show()

# 10. Monthly Average Returns
preprocessed_df['month'] = preprocessed_df['date'].dt.to_period('M')
monthly_returns = preprocessed_df.groupby('month')['daily_return'].mean()

plt.figure(figsize=(15, 6))
monthly_returns.plot(kind='bar')
plt.title('Average Monthly Returns')
plt.xlabel('Month')
plt.ylabel('Average Daily Return (%)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Save preprocessed data
print("\nSaving preprocessed data...")
preprocessed_df.to_csv('preprocessed_stock_data.csv', index=False)
print("Preprocessed data saved to 'preprocessed_stock_data.csv'") 