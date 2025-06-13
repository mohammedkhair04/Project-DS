import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set the style for better visualizations
plt.style.use('seaborn')
sns.set_palette("husl")

def load_and_prepare_data():
    # Load the data
    df = pd.read_csv('all_stocks_5yr.csv')
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Calculate daily returns
    df['daily_return'] = df.groupby('Name')['close'].pct_change() * 100
    
    return df

def plot_stock_price_trends(df, stock_names=None, start_date=None, end_date=None):
    """
    Plot price trends for selected stocks
    """
    if stock_names is None:
        stock_names = ['AAPL', 'MSFT', 'GOOGL']  # Default stocks
    
    plt.figure(figsize=(15, 8))
    
    for stock in stock_names:
        stock_data = df[df['Name'] == stock]
        if start_date:
            stock_data = stock_data[stock_data['date'] >= start_date]
        if end_date:
            stock_data = stock_data[stock_data['date'] <= end_date]
        
        plt.plot(stock_data['date'], stock_data['close'], label=stock)
    
    plt.title('Stock Price Trends Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_volume_analysis(df, stock_name):
    """
    Plot trading volume analysis for a specific stock
    """
    stock_data = df[df['Name'] == stock_name]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Volume over time
    ax1.plot(stock_data['date'], stock_data['volume'])
    ax1.set_title(f'Trading Volume for {stock_name}')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Volume')
    ax1.grid(True)
    ax1.tick_params(axis='x', rotation=45)
    
    # Volume distribution
    sns.histplot(data=stock_data, x='volume', bins=50, ax=ax2)
    ax2.set_title(f'Volume Distribution for {stock_name}')
    ax2.set_xlabel('Volume')
    ax2.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

def plot_price_distribution(df, stock_name):
    """
    Plot price distribution analysis for a specific stock
    """
    stock_data = df[df['Name'] == stock_name]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Price distribution
    sns.histplot(data=stock_data, x='close', bins=50, ax=ax1)
    ax1.set_title(f'Price Distribution for {stock_name}')
    ax1.set_xlabel('Price')
    ax1.set_ylabel('Frequency')
    
    # Daily returns distribution
    sns.histplot(data=stock_data, x='daily_return', bins=50, ax=ax2)
    ax2.set_title(f'Daily Returns Distribution for {stock_name}')
    ax2.set_xlabel('Daily Return (%)')
    ax2.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df, stock_names=None):
    """
    Plot correlation matrix between different stocks
    """
    if stock_names is None:
        stock_names = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'FB']
    
    # Create a pivot table with closing prices
    pivot_df = df[df['Name'].isin(stock_names)].pivot(
        index='date', columns='Name', values='close'
    )
    
    # Calculate correlation matrix
    corr_matrix = pivot_df.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Stock Price Correlation Matrix')
    plt.tight_layout()
    plt.show()

def plot_volatility_analysis(df, stock_name):
    """
    Plot volatility analysis for a specific stock
    """
    stock_data = df[df['Name'] == stock_name].copy()
    
    # Calculate rolling volatility (20-day window)
    stock_data['volatility'] = stock_data['daily_return'].rolling(window=20).std()
    
    plt.figure(figsize=(15, 8))
    plt.plot(stock_data['date'], stock_data['volatility'])
    plt.title(f'20-Day Rolling Volatility for {stock_name}')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Example usage of visualization functions
    plot_stock_price_trends(df, ['AAPL', 'MSFT', 'GOOGL'])
    plot_volume_analysis(df, 'AAPL')
    plot_price_distribution(df, 'AAPL')
    plot_correlation_matrix(df)
    plot_volatility_analysis(df, 'AAPL')

if __name__ == "__main__":
    main()