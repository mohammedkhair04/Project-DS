import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
import os
import warnings
warnings.filterwarnings('ignore')

def optimize_dataframe(df):
    """Optimize DataFrame memory usage."""
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif df[col].dtype == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
    return df

def process_chunk(chunk, required_cols=['close', 'high', 'low', 'volume']):
    """Process a chunk of data with optimized calculations."""
    try:
        # Select only required columns to reduce memory usage
        chunk = chunk[required_cols]
        
        # Convert to numpy arrays for faster computation
        close_array = chunk['close'].values
        volume_array = chunk['volume'].values
        volatility = chunk['high'].values - chunk['low'].values
        
        result = {
            'mean_close': np.nanmean(close_array),  # Handle NaN values
            'mean_volume': np.nanmean(volume_array),
            'volatility': pd.Series(volatility)
        }
        return result
    except Exception as e:
        print(f"Error in chunk processing: {str(e)}")
        return None

def parallel_process_data(df):
    """Process the dataframe using optimized parallel processing."""
    try:
        # Optimize DataFrame memory usage
        df = optimize_dataframe(df)
        
        # Use optimal number of processes
        num_processes = max(1, cpu_count() // 2)  # Use half of available cores
        print(f'Using {num_processes} processes for parallel processing')
        
        # Calculate optimal chunk size (aim for fewer, larger chunks)
        total_rows = len(df)
        chunk_size = max(10000, total_rows // (num_processes * 2))
        print(f'Processing {total_rows} rows in chunks of {chunk_size}')
        
        # Create chunks with minimal memory overhead
        chunks = [df.iloc[i:i + chunk_size] for i in range(0, total_rows, chunk_size)]
        
        # Process data in parallel with progress tracking
        with Pool(processes=num_processes) as pool:
            results = []
            for i, result in enumerate(pool.imap(process_chunk, chunks)):
                if result:
                    results.append(result)
                print(f'Processed chunk {i+1}/{len(chunks)}', end='\r')
        
        if not results:
            raise ValueError("No results were processed successfully")
        
        # Combine results efficiently
        final_results = {
            'mean_close': np.mean([r['mean_close'] for r in results]),
            'mean_volume': np.mean([r['mean_volume'] for r in results]),
            'volatility': pd.concat([r['volatility'] for r in results], ignore_index=True)
        }
        
        print('\nProcessing completed successfully')
        return final_results
    
    except Exception as e:
        print(f"Error in parallel processing: {str(e)}")
        return None