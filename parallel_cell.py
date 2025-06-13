# Parallel Processing Implementation
from multiprocessing import Pool
import os

# Define number of processes based on CPU cores
num_processes = os.cpu_count()

# Function to process data chunks
def process_chunk(chunk):
    result = {
        'mean_close': chunk['close'].mean(),
        'mean_volume': chunk['volume'].mean(),
        'volatility': chunk['high'] - chunk['low']
    }
    return result

# Split data into chunks for parallel processing
chunk_size = len(df) // num_processes
chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

# Process data in parallel
with Pool(processes=num_processes) as pool:
    results = pool.map(process_chunk, chunks)

# Combine results
final_results = {
    'mean_close': np.mean([r['mean_close'] for r in results]),
    'mean_volume': np.mean([r['mean_volume'] for r in results]),
    'volatility': pd.concat([r['volatility'] for r in results])
}

# Display results
print("Parallel Processing Results:")
print(f"Mean Close Price: {final_results['mean_close']:.2f}")
print(f"Mean Volume: {final_results['mean_volume']:.2f}")
print("\nVolatility Statistics:")
print(final_results['volatility'].describe())

# Visualize volatility distribution
plt.figure(figsize=(12, 6))
sns.histplot(data=final_results['volatility'], bins=50)
plt.title('Distribution of Stock Price Volatility')
plt.xlabel('Volatility (High - Low)')
plt.ylabel('Count')
plt.show()