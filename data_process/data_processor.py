# data_processor.py

# 원하는 분봉으로 변경하세요 (예: '1min', '3min', '5min', '15min', '30min', '1h')
TARGET_TIMEFRAME = '3min'

import pandas as pd
import numpy as np
from typing import Tuple
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Starting basic data cleaning.")
    
    if df.empty:
        logging.warning("Input DataFrame is empty. Returning as is.")
        return df

    cleaned_df = df.copy()
    cleaned_df.columns = [col.lower() for col in cleaned_df.columns]

    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in cleaned_df.columns for col in required_columns):
        raise ValueError(f"Input DataFrame must contain all required columns: {required_columns}")

    if cleaned_df.index.tz is None:
        cleaned_df = cleaned_df.tz_localize('UTC')
    else:
        cleaned_df = cleaned_df.tz_convert('UTC')

    cleaned_df.sort_index(inplace=True)
    cleaned_df = cleaned_df[~cleaned_df.index.duplicated(keep='last')]

    for col in required_columns:
        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')

    original_rows = len(cleaned_df)
    cleaned_df = cleaned_df[
        (cleaned_df['open'] > 0) & (cleaned_df['high'] > 0) &
        (cleaned_df['low'] > 0) & (cleaned_df['close'] > 0) &
        (cleaned_df['volume'] >= 0)
    ]
    logging.info(f"Removed {original_rows - len(cleaned_df)} rows with non-positive price/volume.")

    original_rows = len(cleaned_df)
    cleaned_df.dropna(inplace=True)
    logging.info(f"Removed {original_rows - len(cleaned_df)} rows with NaN values.")

    if cleaned_df.empty:
        raise ValueError("DataFrame became empty after cleaning. Check raw data quality.")

    logging.info("Basic data cleaning finished.")
    return cleaned_df

def logical_filter(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    logging.info("Applying logical filter (low <= open/close <= high).")
    
    original_rows = len(df)
    filtered_df = df[
        (df['low'] <= df['open']) & (df['low'] <= df['close']) &
        (df['high'] >= df['open']) & (df['high'] >= df['close'])
    ].copy()

    removed_count = original_rows - len(filtered_df)
    logging.info(f"Removed {removed_count} rows that violated logical constraints.")
    
    return filtered_df, removed_count

def spike_filter(df: pd.DataFrame, threshold: float = 0.20) -> Tuple[pd.DataFrame, int]:
    logging.info(f"Applying spike filter with threshold {threshold:.2%}.")
    
    original_rows = len(df)
    returns = df['close'].pct_change().abs()
    filtered_df = df[returns <= threshold].copy()
    removed_count = original_rows - len(filtered_df)

    if removed_count > 0:
        removed_timestamps = df.index.difference(filtered_df.index)
        logging.warning(f"Removed {removed_count} spike candles. Timestamps: {removed_timestamps.to_list()}")
    else:
        logging.info("No spike candles found.")

    return filtered_df, removed_count

def resample_ohlcv(df: pd.DataFrame, out_freq: str) -> pd.DataFrame:
    logging.info(f"Resampling data to '{out_freq}'.")

    resampling_rules = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }

    resampled_df = df.resample(out_freq).apply(resampling_rules).dropna()
    logging.info(f"Resampling complete. Result has {len(resampled_df)} rows.")
    
    return resampled_df

def process_raw_to_timeframe(raw_csv_path: str, processed_csv_path: str) -> pd.DataFrame:
    logging.info(f"Processing {raw_csv_path} to {TARGET_TIMEFRAME} bars")
    
    raw_df = pd.read_csv(raw_csv_path, index_col=0, parse_dates=True)
    logging.info(f"Loaded {len(raw_df)} rows from raw CSV")
    
    cleaned_df = basic_clean(raw_df)
    filtered_df, removed_logical = logical_filter(cleaned_df)
    spike_filtered_df, removed_spikes = spike_filter(filtered_df, threshold=0.20)
    resampled_df = resample_ohlcv(spike_filtered_df, TARGET_TIMEFRAME)
    
    output_path = Path(processed_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    resampled_df.to_csv(output_path, index=True)
    logging.info(f"Saved {len(resampled_df)} {TARGET_TIMEFRAME} bars to {output_path}")
    
    return resampled_df

if __name__ == '__main__':
    source = "binance"
    symbol_name = "BTCUSDT"
    interval_str = "1m"
    
    raw_csv_path = Path(f"../data/raw/{source}_{symbol_name}_{interval_str}_raw.csv")
    processed_csv_path = Path(f"../data/processed/{source}_{symbol_name}_{TARGET_TIMEFRAME}_processed.csv")
    
    if raw_csv_path.exists():
        print(f"\n--- Processing {raw_csv_path} to {TARGET_TIMEFRAME} bars ---")
        processed_df = process_raw_to_timeframe(str(raw_csv_path), str(processed_csv_path))
        
        print(f"\n--- Processing Results ---")
        print(f"Original 1-minute bars: {len(pd.read_csv(raw_csv_path))}")
        print(f"Processed {TARGET_TIMEFRAME} bars: {len(processed_df)}")
        print(f"\nFirst 5 rows of processed data:")
        print(processed_df.head())
        
        if Path(processed_csv_path).exists():
            verification_df = pd.read_csv(processed_csv_path, index_col=0, parse_dates=True)
            print(f"Verification: CSV file contains {len(verification_df)} rows")
    else:
        print(f"Raw CSV file not found: {raw_csv_path}")
        print("Please run data_loader.py first to generate raw data.")