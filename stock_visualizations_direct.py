import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set style for better visualizations
plt.style.use('seaborn')
sns.set_palette("husl")

# Load the data
df = pd.read_csv('all_stocks_5yr.csv')
df['date'] = pd.to_datetime(df['date'])

# Calculate daily returns
df['daily_return'] = df.groupby('Name')['close'].pct_change() * 100

# 1. Stock Price Trends for Top 5 Stocks
plt.figure(figsize=(15, 8))
top_stocks = df['Name'].value_counts().head(5).index
for stock in top_stocks:
    stock_data = df[df['Name'] == stock]
    plt.plot(stock_data['date'], stock_data['close'], label=stock)
plt.title('Top 5 Stocks Price Trends')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Candlestick-like Plot for a Single Stock (e.g., AAPL)
stock_data = df[df['Name'] == 'AAPL'].copy()
plt.figure(figsize=(15, 8))
plt.plot(stock_data['date'], stock_data['close'], label='Close Price', color='blue')
plt.fill_between(stock_data['date'], 
                 stock_data['low'], 
                 stock_data['high'], 
                 alpha=0.3, 
                 color='gray', 
                 label='Price Range')
plt.title('AAPL Stock Price with Range')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Volume Analysis
plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
plt.plot(stock_data['date'], stock_data['volume'], color='green')
plt.title('AAPL Trading Volume Over Time')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.grid(True)
plt.xticks(rotation=45)

plt.subplot(2, 1, 2)
sns.histplot(data=stock_data, x='volume', bins=50, color='green')
plt.title('Volume Distribution')
plt.xlabel('Volume')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 4. Price and Returns Distribution
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
sns.histplot(data=stock_data, x='close', bins=50, color='blue')
plt.title('Price Distribution')
plt.xlabel('Price')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.histplot(data=stock_data, x='daily_return', bins=50, color='red')
plt.title('Daily Returns Distribution')
plt.xlabel('Daily Return (%)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 5. Moving Averages
stock_data['MA20'] = stock_data['close'].rolling(window=20).mean()
stock_data['MA50'] = stock_data['close'].rolling(window=50).mean()

plt.figure(figsize=(15, 8))
plt.plot(stock_data['date'], stock_data['close'], label='Close Price', alpha=0.7)
plt.plot(stock_data['date'], stock_data['MA20'], label='20-day MA', color='red')
plt.plot(stock_data['date'], stock_data['MA50'], label='50-day MA', color='green')
plt.title('AAPL Price with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 6. Volatility Analysis
stock_data['volatility'] = stock_data['daily_return'].rolling(window=20).std()

plt.figure(figsize=(15, 8))
plt.plot(stock_data['date'], stock_data['volatility'], color='purple')
plt.title('20-Day Rolling Volatility')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 7. Correlation Heatmap for Top 5 Stocks
top_stocks_data = df[df['Name'].isin(top_stocks)].pivot(
    index='date', 
    columns='Name', 
    values='close'
)
correlation_matrix = top_stocks_data.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            fmt='.2f')
plt.title('Stock Price Correlation Matrix')
plt.tight_layout()
plt.show()

# 8. Box Plot of Daily Returns
plt.figure(figsize=(15, 6))
sns.boxplot(data=df[df['Name'].isin(top_stocks)], 
            x='Name', 
            y='daily_return')
plt.title('Daily Returns Distribution by Stock')
plt.xlabel('Stock')
plt.ylabel('Daily Return (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 9. Scatter Plot of Volume vs Price
plt.figure(figsize=(12, 8))
plt.scatter(stock_data['volume'], 
           stock_data['close'], 
           alpha=0.5, 
           c=stock_data['daily_return'], 
           cmap='viridis')
plt.colorbar(label='Daily Return (%)')
plt.title('Volume vs Price (colored by Daily Return)')
plt.xlabel('Volume')
plt.ylabel('Price (USD)')
plt.grid(True)
plt.tight_layout()
plt.show()

# 10. Monthly Average Returns
stock_data['month'] = stock_data['date'].dt.to_period('M')
monthly_returns = stock_data.groupby('month')['daily_return'].mean()

plt.figure(figsize=(15, 6))
monthly_returns.plot(kind='bar')
plt.title('Average Monthly Returns')
plt.xlabel('Month')
plt.ylabel('Average Daily Return (%)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show() 