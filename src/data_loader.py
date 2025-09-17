import pandas as pd
import numpy as np
from datetime import datetime
import logging

class DataLoader:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def process_ohlcv(self, df, interval_minutes=5, method='drop'):
        """Process OHLCV data with all validation and cleaning steps"""
        
        # 1. Column standardization
        df.columns = df.columns.str.lower()
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {required_cols}")
        
        df = df[required_cols].copy()
        
        # 2. Time processing
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.sort_values('timestamp').drop_duplicates('timestamp', keep='last')
        
        # 3. Data type validation
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        
        # Remove negative volume and non-positive prices
        df = df[(df['volume'] >= 0) & 
                (df[['open', 'high', 'low', 'close']] > 0).all(axis=1)]
        
        # 4. Remove missing values
        df = df.dropna()
        
        # 5. Price logic validation
        valid_price = ((df['low'] <= df['open']) & (df['open'] <= df['high']) &
                      (df['low'] <= df['close']) & (df['close'] <= df['high']))
        df = df[valid_price]
        
        # 6. Spike removal (>20% returns)
        df = df.sort_values('timestamp')
        returns = df['close'].pct_change().abs()
        spike_mask = returns > 0.2
        spike_count = spike_mask.sum()
        if spike_count > 0:
            self.logger.info(f"Removed {spike_count} spike data points")
        df = df[~spike_mask]
        
        # 7. Continuity check
        if len(df) > 0:
            start_time = df['timestamp'].min()
            end_time = df['timestamp'].max()
            expected_index = pd.date_range(start_time, end_time, 
                                         freq=f'{interval_minutes}min', tz='UTC')
            missing_count = len(expected_index) - len(df)
            
            if missing_count > 0:
                self.logger.info(f"Missing {missing_count} candles in time range")
                
                if method == 'drop':
                    # Keep only existing data
                    pass
                elif method == 'ffill':
                    # Forward fill missing values
                    df = df.set_index('timestamp').reindex(expected_index).fillna(method='ffill')
                    df = df.reset_index().rename(columns={'index': 'timestamp'})
                elif method == 'rest':
                    self.logger.warning("REST re-request needed for missing data")
        
        return df.reset_index(drop=True)