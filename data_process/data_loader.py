# data_loader.py

import pandas as pd
from pathlib import Path
from typing import Union
from data_fetcher import fetch_klines_binance, Client
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_raw_csv(path: Union[str, Path]) -> pd.DataFrame:
    logging.info(f"Loading raw CSV from: {path}")
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"No file found at the specified path: {filepath}")

    try:
        df = pd.read_csv(filepath)

        df.columns = [col.lower() for col in df.columns]

        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in df.columns):
            raise ValueError(f"CSV file must contain all required columns: {required_columns}")

        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)

        df.sort_values(by='timestamp', inplace=True)
        df = df.drop_duplicates(subset=['timestamp'], keep='first')

        df.set_index('timestamp', inplace=True)

        logging.info(f"Successfully loaded and cleaned {len(df)} rows from {path}")
        return df

    except Exception as e:
        logging.error(f"Failed to load or process CSV file at {path}: {e}")
        raise

def save_raw_csv(df: pd.DataFrame, path: Union[str, Path], mode: str = "w"):
    logging.info(f"Saving DataFrame to CSV at: {path} (mode: {'append' if mode=='a' else 'overwrite'})")

    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if mode == "a" and filepath.exists():
        df.to_csv(filepath, mode="a", header=False, index=True)
    else:
        df.to_csv(filepath, mode="w", header=True, index=True)

    logging.info(f"Save operation completed successfully.")

def save_processed_parquet(df: pd.DataFrame, path: Union[str, Path]):
    logging.info(f"Saving processed DataFrame to Parquet at: {path}")
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(filepath, index=True)
    logging.info(f"Parquet save operation completed successfully.")

def build_from_fetch(symbol: str, interval: str, start_str: str, end_str: str, out_csv_path: Union[str, Path]) -> Path:
    logging.info(f"Building data for {symbol} interval {interval}...")

    out_filepath = Path(out_csv_path)
    existing_df = None

    if out_filepath.exists():
        logging.info(f"Existing file found at {out_filepath}. Checking for new data...")
        try:
            existing_df = load_raw_csv(out_filepath)

            if not existing_df.empty:
                last_timestamp = existing_df.index.max()
                start_str = (last_timestamp + pd.Timedelta(minutes=1)).isoformat()
                logging.info(f"Fetching new data from {start_str}...")
            else:
                logging.info(f"Existing file is empty. Fetching all data from {start_str}.")
                existing_df = None
        except Exception as e:
            logging.warning(f"Failed to load existing file, reprocessing all. Error: {e}")
            existing_df = None
    else:
        logging.info(f"No existing file. Fetching all data from {start_str}.")
        existing_df = None

    fetched_df = fetch_klines_binance(symbol, interval, start_str, end_str)

    if existing_df is not None:
        combined_df = pd.concat([existing_df, fetched_df])
    else:
        combined_df = fetched_df

    if not combined_df.empty:
        combined_df.reset_index(inplace=True)
        combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='last')
        combined_df.sort_values(by='timestamp', inplace=True)
        combined_df.set_index('timestamp', inplace=True)

    save_raw_csv(combined_df, out_filepath, mode="w")

    if not combined_df.empty:
        logging.info(f"Data build process completed. Total {len(combined_df)} rows. Last timestamp {combined_df.index.max()}")
    else:
        logging.info("Data build process completed. No data found or saved.")

    return out_filepath

if __name__ == '__main__':
    CURRENT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = CURRENT_DIR.parent

    source = "binance"
    symbol_name = "BTCUSDT"
    interval_str = "1m"
    interval_client = Client.KLINE_INTERVAL_1MINUTE

    initial_start_date = "2025-01-01"

    raw_data_dir = PROJECT_ROOT / "data" / "raw"
    processed_data_dir = PROJECT_ROOT / "data" / "processed"

    raw_filename = f"{source}_{symbol_name}_{interval_str}_raw.csv"
    csv_path = raw_data_dir / raw_filename

    processed_filename = f"{source}_{symbol_name}_{interval_str}_processed.parquet"
    parquet_path = processed_data_dir / processed_filename

    print("\n--- Building data from fetcher (Incremental Update) ---")
    build_from_fetch(
        symbol=symbol_name,
        interval=interval_client,
        start_str=initial_start_date,
        end_str=None,
        out_csv_path=csv_path
    )

    print("\n--- Loading and verifying the raw CSV file ---")
    loaded_df = load_raw_csv(csv_path)
    print("CSV Tail (latest 5 rows):")
    print(loaded_df.tail())

    print("\n--- Saving data to Parquet format ---")
    save_processed_parquet(loaded_df, parquet_path)
    print(f"Data saved to Parquet format at {parquet_path}")

    print("\n--- Verifying the Parquet file ---")
    parquet_df = pd.read_parquet(parquet_path)
    print("Parquet Tail (latest 5 rows):")
    print(parquet_df.tail())