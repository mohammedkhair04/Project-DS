import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# Set style for better visualizations
plt.style.use('seaborn')
sns.set_palette("husl")

# Load and preprocess data
print("Loading and preprocessing data...")
df = pd.read_csv('all_stocks_5yr.csv')
df['date'] = pd.to_datetime(df['date'])

# Basic data cleaning
df = df.dropna()

# Convert stock names to binary
unique_stocks = df['Name'].unique()
stock_to_number = {stock: i for i, stock in enumerate(unique_stocks)}
df['Name_binary'] = df['Name'].map(stock_to_number)

# Scale numerical features
numeric_columns = ['open', 'high', 'low', 'close', 'volume']
scaler = MinMaxScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Calculate additional features
df['daily_return'] = df.groupby('Name')['close'].pct_change() * 100
df['volatility'] = df.groupby('Name')['daily_return'].rolling(window=20).std().reset_index(0, drop=True)

# Save preprocessed data
df.to_csv('preprocessed_data.csv', index=False)
print("Preprocessed data saved to 'preprocessed_data.csv'")

# Create visualizations
print("\nCreating visualizations...")

# 1. Correlation Matrix
plt.figure(figsize=(12, 8))
correlation_matrix = df[numeric_columns + ['daily_return']].corr()
sns.heatmap(correlation_matrix, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            fmt='.2f')
plt.title('Correlation Matrix of Scaled Features')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# 2. Stock Name Distribution
plt.figure(figsize=(15, 6))
sns.countplot(data=df, x='Name_binary')
plt.title('Distribution of Stock Names (Numeric Encoding)')
plt.xlabel('Numeric Stock Code')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('stock_distribution.png')
plt.close()

# 3. Price Distribution by Stock
plt.figure(figsize=(15, 6))
sns.boxplot(data=df, x='Name_binary', y='close')
plt.title('Scaled Price Distribution by Stock')
plt.xlabel('Numeric Stock Code')
plt.ylabel('Scaled Price')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('price_distribution.png')
plt.close()

# 4. Volume vs Price Scatter
plt.figure(figsize=(12, 8))
plt.scatter(df['volume'], 
           df['close'], 
           alpha=0.5, 
           c=df['daily_return'], 
           cmap='viridis')
plt.colorbar(label='Daily Return (%)')
plt.title('Scaled Volume vs Price (colored by Daily Return)')
plt.xlabel('Scaled Volume')
plt.ylabel('Scaled Price')
plt.grid(True)
plt.tight_layout()
plt.savefig('volume_price_scatter.png')
plt.close()

# 5. Volatility Distribution
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='volatility', bins=50)
plt.title('Distribution of 20-Day Rolling Volatility')
plt.xlabel('Volatility')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('volatility_distribution.png')
plt.close()

# 6. Daily Returns Distribution
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='daily_return', bins=50)
plt.title('Distribution of Daily Returns')
plt.xlabel('Daily Return (%)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('returns_distribution.png')
plt.close()

# 7. Price Trends for Top 5 Stocks
top_stocks = df['Name'].value_counts().head(5).index
plt.figure(figsize=(15, 8))
for stock in top_stocks:
    stock_data = df[df['Name'] == stock]
    plt.plot(stock_data['date'], stock_data['close'], label=stock)
plt.title('Scaled Price Trends for Top 5 Stocks')
plt.xlabel('Date')
plt.ylabel('Scaled Price')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('price_trends.png')
plt.close()

# 8. Volume Trends
plt.figure(figsize=(15, 6))
for stock in top_stocks:
    stock_data = df[df['Name'] == stock]
    plt.plot(stock_data['date'], stock_data['volume'], label=stock, alpha=0.7)
plt.title('Scaled Volume Trends for Top 5 Stocks')
plt.xlabel('Date')
plt.ylabel('Scaled Volume')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('volume_trends.png')
plt.close()

# 9. Price vs Volatility
plt.figure(figsize=(12, 8))
plt.scatter(df['close'], 
           df['volatility'], 
           alpha=0.5, 
           c=df['daily_return'], 
           cmap='viridis')
plt.colorbar(label='Daily Return (%)')
plt.title('Price vs Volatility (colored by Daily Return)')
plt.xlabel('Scaled Price')
plt.ylabel('Volatility')
plt.grid(True)
plt.tight_layout()
plt.savefig('price_volatility.png')
plt.close()

# 10. Monthly Average Returns
df['month'] = df['date'].dt.to_period('M')
monthly_returns = df.groupby('month')['daily_return'].mean()

plt.figure(figsize=(15, 6))
monthly_returns.plot(kind='bar')
plt.title('Average Monthly Returns')
plt.xlabel('Month')
plt.ylabel('Average Daily Return (%)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig('monthly_returns.png')
plt.close()

# Save the stock name mapping
mapping_df = pd.DataFrame({
    'Stock_Name': list(stock_to_number.keys()),
    'Binary_Code': list(stock_to_number.values())
})
mapping_df.to_csv('stock_name_mapping.csv', index=False)

# Print summary information
print("\nAnalysis Summary:")
print(f"Number of unique stocks: {len(stock_to_number)}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Number of data points: {len(df)}")
print("\nVisualizations have been saved as PNG files:")
print("1. correlation_matrix.png")
print("2. stock_distribution.png")
print("3. price_distribution.png")
print("4. volume_price_scatter.png")
print("5. volatility_distribution.png")
print("6. returns_distribution.png")
print("7. price_trends.png")
print("8. volume_trends.png")
print("9. price_volatility.png")
print("10. monthly_returns.png") 