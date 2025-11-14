# data_processor.py

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

def process_raw_to_timeframe(raw_csv_path: str, processed_csv_path: str, timeframe: str) -> pd.DataFrame:
    logging.info(f"Processing {raw_csv_path} to {timeframe} bars, appending to {processed_csv_path}")

    processed_path = Path(processed_csv_path)
    existing_processed_df = None
    last_processed_timestamp = None

    if processed_path.exists():
        try:
            existing_processed_df = pd.read_csv(processed_path, index_col=0, parse_dates=True)
            if not isinstance(existing_processed_df.index, pd.DatetimeIndex):
                existing_processed_df.index = pd.to_datetime(existing_processed_df.index, errors='coerce')
                existing_processed_df.dropna(subset=[existing_processed_df.index.name], inplace=True)

            if existing_processed_df.index.tz is None:
                existing_processed_df = existing_processed_df.tz_localize('UTC')
            else:
                existing_processed_df = existing_processed_df.tz_convert('UTC')

            if not existing_processed_df.empty:
                last_processed_timestamp = existing_processed_df.index.max()
                logging.info(f"Existing processed data found. Last timestamp: {last_processed_timestamp}")
            else:
                existing_processed_df = None
        except Exception as e:
            logging.warning(f"Could not load or parse existing processed file {processed_path}, reprocessing all. Error: {e}")
            existing_processed_df = None
            last_processed_timestamp = None
    else:
        logging.info(f"Processed file {processed_path} not found. Processing from scratch.")

    try:
        raw_df = pd.read_csv(raw_csv_path, index_col='timestamp', parse_dates=True)
        if raw_df.index.tz is None:
            raw_df = raw_df.tz_localize('UTC')
        else:
            raw_df = raw_df.tz_convert('UTC')
        logging.info(f"Loaded {len(raw_df)} rows from raw CSV {raw_csv_path}")
    except FileNotFoundError:
        logging.error(f"Raw CSV file not found: {raw_csv_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading raw CSV {raw_csv_path}: {e}")
        raise

    df_to_process = None
    if last_processed_timestamp:
        df_to_process = raw_df[raw_df.index >= last_processed_timestamp].copy()
        logging.info(f"Processing raw data from {last_processed_timestamp} onwards. Found {len(df_to_process)} rows.")
    else:
        df_to_process = raw_df.copy()
        logging.info("Processing all raw data.")

    if df_to_process.empty or len(df_to_process) <= 1:
        logging.info("No new raw data to process or not enough data for filtering.")
        return existing_processed_df if existing_processed_df is not None else pd.DataFrame()

    cleaned_df = basic_clean(df_to_process)
    filtered_df, _ = logical_filter(cleaned_df)
    spike_filtered_df, _ = spike_filter(filtered_df, threshold=0.20)
    resampled_df = resample_ohlcv(spike_filtered_df, timeframe)

    if resampled_df.empty:
        logging.info("Processing resulted in an empty DataFrame.")
        return existing_processed_df if existing_processed_df is not None else pd.DataFrame()

    if existing_processed_df is not None:
        final_processed_df = pd.concat([existing_processed_df, resampled_df])
        final_processed_df = final_processed_df[~final_processed_df.index.duplicated(keep='last')]
    else:
        final_processed_df = resampled_df

    final_processed_df.sort_index(inplace=True)

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    final_processed_df.to_csv(processed_path, index=True)
    logging.info(f"Saved {len(final_processed_df)} {timeframe} bars to {processed_path}")

    return final_processed_df

# --- 예제 코드 실행 부분 ---
if __name__ == '__main__':
    # 이 파일(data_processor.py)이 있는 폴더 (Project/data_process)
    CURRENT_DIR = Path(__file__).resolve().parent
    # 프로젝트 루트 폴더 (Project/)
    PROJECT_ROOT = CURRENT_DIR.parent

    TARGET_TIMEFRAME = '3min'

    source = "binance"
    symbol_name = "BTCUSDT"
    raw_interval_str = "1m"

    # 원본 CSV 파일 경로 (프로젝트 루트/data/raw)
    raw_csv_path = PROJECT_ROOT / "data" / "raw" / f"{source}_{symbol_name}_{raw_interval_str}_raw.csv"
    # 최종 처리 결과를 저장할 CSV 파일 경로 (프로젝트 루트/data/processed)
    processed_csv_path = PROJECT_ROOT / "data" / "processed" / f"{source}_{symbol_name}_{TARGET_TIMEFRAME}_processed.csv"

    if raw_csv_path.exists():
        print(f"\n--- Processing {raw_csv_path} to {TARGET_TIMEFRAME} bars ---")
        processed_df = process_raw_to_timeframe(str(raw_csv_path), str(processed_csv_path), TARGET_TIMEFRAME)

        print(f"\n--- Processing Results ---")
        try:
            raw_row_count = len(pd.read_csv(raw_csv_path))
            print(f"Original {raw_interval_str} bars: {raw_row_count}")
        except Exception:
            print(f"Could not read original file length.")

        print(f"Processed {TARGET_TIMEFRAME} bars: {len(processed_df)}")
        print(f"\nLast 5 rows of processed data:")
        print(processed_df.tail())

        if Path(processed_csv_path).exists():
            try:
                verification_df = pd.read_csv(processed_csv_path, index_col=0, parse_dates=True)
                print(f"Verification: Saved CSV file contains {len(verification_df)} rows")
            except Exception as e:
                print(f"Verification failed: Could not read saved file. Error: {e}")
    else:
        print(f"Raw CSV file not found: {raw_csv_path}")
        print("Please run data_loader.py first to generate raw data.")