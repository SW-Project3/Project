# data_loader.py

# --- 라이브러리 불러오기 ---
# 'pandas'는 데이터를 표(DataFrame) 형태로 다루는 라이브러리입니다.
import pandas as pd
# 'pathlib'은 어떤 운영체제에서든 파일 경로를 쉽고 안전하게 다룰 수 있게 도와주는 라이브러리입니다.
from pathlib import Path
# 'typing'은 코드의 변수나 함수의 타입(형식)을 명시하여 가독성과 안정성을 높여주는 도구입니다.
from typing import Union
# 이전에 우리가 만든 data_fetcher.py 파일에서 과거 데이터 수집 함수와 Client 객체를 가져옵니다.
from data_fetcher import fetch_klines_binance, Client
# 'logging'은 프로그램의 실행 상태나 오류를 기록(로그)으로 남기기 위한 라이브러리입니다.
import logging
# 'os'는 운영체제와 상호작용하기 위한 라이브러리로, 여기서는 파일 경로를 다루기 위해 사용합니다.
import os

# --- 기본 설정 ---
# 로그를 어떤 형식으로, 어떤 수준(INFO 이상)까지 출력할지 기본 설정을 합니다.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. Raw CSV 로드 및 최소 정리 함수 ---
# load_raw_csv 함수를 정의합니다. CSV 파일을 불러와 기본적인 정리를 수행합니다.
def load_raw_csv(path: Union[str, Path]) -> pd.DataFrame:
    # 함수의 기능, 입력값, 결과값에 대한 상세 설명(Docstring)입니다.
    """
    raw CSV 파일을 로드하고 최소한의 정리를 수행합니다.
    (컬럼 소문자화, UTC DatetimeIndex 설정, 정렬, 중복 제거)
    """
    # "어떤 파일 경로에서 CSV를 불러옵니다" 라는 로그를 남깁니다.
    logging.info(f"Loading raw CSV from: {path}")

    # Path 객체로 변환하여 파일 경로를 다루기 쉽게 만듭니다.
    filepath = Path(path)
    # 만약 파일이 존재하지 않는다면,
    if not filepath.exists():
        # "파일을 찾을 수 없음" 이라는 에러를 발생시켜 프로그램을 중단합니다.
        raise FileNotFoundError(f"No file found at the specified path: {filepath}")

    # try-except 구문: 파일 읽기 및 처리 중 에러가 발생하면 처리합니다.
    try:
        # pandas의 read_csv 함수로 CSV 파일을 읽어 DataFrame으로 만듭니다.
        df = pd.read_csv(filepath)

        # DataFrame의 모든 컬럼(열) 이름을 소문자로 변경합니다. (예: 'Open' -> 'open')
        df.columns = [col.lower() for col in df.columns]

        # 필수 컬럼 목록을 정의합니다.
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        # 만약 필수 컬럼 중 하나라도 df에 존재하지 않는다면,
        if not all(col in df.columns for col in df.columns):
            # "필수 컬럼이 누락됨" 이라는 에러를 발생시켜 프로그램을 중단합니다.
            raise ValueError(f"CSV file must contain all required columns: {required_columns}")

        # 'timestamp' 열의 데이터를 날짜/시간 형식으로 변환합니다. (에러 발생 시 무시)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        # 변환에 실패하여 비어있는 행(NaT)이 있다면 모두 제거합니다.
        df.dropna(subset=['timestamp'], inplace=True)
        # 'timestamp' 열을 DataFrame의 인덱스로 설정합니다.
        df.set_index('timestamp', inplace=True)

        # 인덱스(시간)가 시간대 정보를 가지고 있지 않다면,
        if df.index.tz is None:
            # UTC 시간대로 설정합니다.
            df = df.tz_localize('UTC')
        # 시간대 정보가 이미 있다면,
        else:
            # UTC 시간대로 변환합니다.
            df = df.tz_convert('UTC')

        # 인덱스(시간)를 기준으로 오름차순으로 정렬합니다.
        df.sort_index(inplace=True)
        # 중복된 인덱스가 있다면 첫 번째 것만 남기고 모두 제거합니다.
        df = df[~df.index.duplicated(keep='first')]

        # "성공적으로 몇 개의 행을 불러왔습니다" 라는 로그를 남깁니다.
        logging.info(f"Successfully loaded and cleaned {len(df)} rows from {path}")
        # 최종적으로 정리된 DataFrame을 반환합니다.
        return df

    # 파일을 읽거나 처리하는 과정에서 에러가 발생하면,
    except Exception as e:
        # 에러 로그를 남기고,
        logging.error(f"Failed to load or process CSV file at {path}: {e}")
        # 프로그램을 중단시킵니다.
        raise

# --- 2. Raw CSV 저장 함수 ---
# save_raw_csv 함수를 정의합니다. DataFrame을 CSV 파일 형태로 저장합니다.
def save_raw_csv(df: pd.DataFrame, path: Union[str, Path], mode: str = "w"):
    """
    DataFrame을 raw CSV 형태로 저장합니다. (기본값: 덮어쓰기)
    """
    # "어떤 파일 경로에 CSV를 저장합니다" 라는 로그를 남깁니다.
    logging.info(f"Saving DataFrame to CSV at: {path} (mode: {'append' if mode=='a' else 'overwrite'})")

    # Path 객체로 변환합니다.
    filepath = Path(path)
    # 저장할 폴더가 존재하지 않는다면, 폴더를 새로 만듭니다.
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # 이어쓰기('a') 모드이고 파일이 이미 존재하는 경우,
    if mode == "a" and filepath.exists():
        # 헤더(컬럼 이름)를 쓰지 않고 데이터만 파일 끝에 추가합니다.
        df.to_csv(filepath, mode="a", header=False, index=True)
    # 그 외의 경우 (덮어쓰기 'w' 모드 또는 파일이 존재하지 않는 경우),
    else:
        # 헤더를 포함하여 새 파일을 쓰거나 기존 파일을 덮어씁니다.
        df.to_csv(filepath, mode="w", header=True, index=True)

    # "저장 완료" 로그를 남깁니다.
    logging.info(f"Save operation completed successfully.")


# --- 3. 처리된 데이터를 Parquet으로 저장하는 함수 ---
# save_processed_parquet 함수를 정의합니다. Parquet은 CSV보다 훨씬 빠르고 용량이 작은 고성용 데이터 형식입니다.
def save_processed_parquet(df: pd.DataFrame, path: Union[str, Path]):
    """
    전처리 완료된 데이터를 Parquet 파일로 저장합니다.
    """
    # "어떤 파일 경로에 Parquet을 저장합니다" 라는 로그를 남깁니다.
    logging.info(f"Saving processed DataFrame to Parquet at: {path}")

    # Path 객체로 변환합니다.
    filepath = Path(path)
    # 저장할 폴더가 존재하지 않는다면, 폴더를 새로 만듭니다.
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # to_parquet 함수로 DataFrame을 Parquet 파일로 저장합니다.
    df.to_parquet(filepath, index=True)

    # "저장 완료" 로그를 남깁니다.
    logging.info(f"Parquet save operation completed successfully.")


# --- 4. 데이터 수집부터 CSV 저장까지 한 번에 처리하는 함수 ---
# build_from_fetch 함수를 정의합니다. 데이터 수집, 정규화, 저장을 한 번에 처리합니다.
def build_from_fetch(symbol: str, interval: str, start_str: str, end_str: str, out_csv_path: Union[str, Path]) -> Path:
    """
    데이터를 fetch하고 정규화한 후, CSV 파일로 저장하는 원샷 함수입니다.
    """
    # "데이터 빌드 프로세스를 시작합니다" 라는 로그를 남깁니다.
    logging.info(f"Building data for {symbol} from {start_str} to {end_str}.")

    # Path 객체로 변환합니다.
    out_filepath = Path(out_csv_path)

    # data_fetcher의 함수를 호출하여 과거 데이터를 가져옵니다.
    fetched_df = fetch_klines_binance(symbol, interval, start_str, end_str)

    # 만약 기존에 저장된 CSV 파일이 있다면,
    if out_filepath.exists():
        # "기존 파일이 존재하여 데이터를 병합합니다" 라는 로그를 남깁니다.
        logging.info(f"Existing file found at {out_filepath}. Merging data.")
        # 기존 CSV 파일을 불러옵니다.
        existing_df = load_raw_csv(out_filepath)
        # 기존 데이터와 새로 가져온 데이터를 위아래로 합칩니다.
        combined_df = pd.concat([existing_df, fetched_df])
        # 합친 데이터에서 중복된 시간의 데이터를 제거합니다. (항상 최신 데이터 유지)
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
    # 기존 파일이 없다면,
    else:
        # 새로 가져온 데이터를 그대로 사용합니다.
        combined_df = fetched_df

    # 시간순으로 정렬합니다.
    combined_df.sort_index(inplace=True)

    # 최종적으로 합쳐지고 정렬된 데이터를 CSV 파일로 저장합니다. (덮어쓰기 모드)
    save_raw_csv(combined_df, out_filepath, mode="w")

    # "데이터 빌드 완료" 로그와 함께 최종 저장 경로를 반환합니다.
    logging.info(f"Data build process completed. Final data saved to {out_filepath}")
    return out_filepath

# --- 예제 코드 실행 부분 ---
# 이 파일을 직접 실행했을 때만 아래 코드가 동작합니다.
if __name__ == '__main__':
    # --- 저장할 정보 및 경로 설정 ---
    # 데이터 소스 이름을 'binance'로 정의합니다.
    source = "binance"
    # 심볼 이름을 API에서 사용하는 대문자 형태('BTCUSDT')로 통일합니다.
    symbol_name = "BTCUSDT"
    # 파일 이름에 사용할 시간 간격 문자열을 '1m'으로 정의합니다.
    interval_str = "1m"
    # API 요청에 사용할 실제 간격 값을 1분봉으로 정합니다.
    interval_client = Client.KLINE_INTERVAL_1MINUTE
    # 요청 시작 날짜를 "2025-01-01"로 정합니다.
    start_date = "2025-01-01"

    # --- 최종 파일 경로 생성 ---
    # 원본 데이터를 저장할 프로젝트 루트의 'data/raw' 폴더 경로를 설정합니다.
    raw_data_dir = Path("../data/raw")
    # 가공된 데이터를 저장할 프로젝트 루트의 'data/processed' 폴더 경로를 설정합니다.
    processed_data_dir = Path("../data/processed")

    # 새로운 파일 이름 규칙에 따라 파일 이름을 생성합니다. (예: binance_BTCUSDT_1m_raw.csv)
    raw_filename = f"{source}_{symbol_name}_{interval_str}_raw.csv"
    # CSV 파일의 최종 저장 경로를 'data/raw' 폴더로 지정합니다.
    csv_path = raw_data_dir / raw_filename

    # Parquet 파일 이름도 새로운 규칙에 따라 생성합니다.
    processed_filename = f"{source}_{symbol_name}_{interval_str}_processed.parquet"
    # Parquet 파일의 최종 저장 경로를 'data/processed' 폴더로 지정합니다.
    parquet_path = processed_data_dir / processed_filename

    # --- 1. build_from_fetch 함수 실행 ---
    # "데이터 빌드 시작" 메시지를 출력합니다.
    print("\n--- Building data from fetcher ---")
    # 위에서 설정한 값들로 데이터를 가져와 CSV 파일로 빌드하는 함수를 호출합니다.
    build_from_fetch(
        symbol=symbol_name,
        interval=interval_client,
        start_str=start_date,
        end_str=None,
        out_csv_path=csv_path
    )

    # --- 2. load_raw_csv 함수 테스트 ---
    # "CSV 파일 확인 시작" 메시지를 출력합니다.
    print("\n--- Loading and verifying the raw CSV file ---")
    # 방금 저장한 CSV 파일을 다시 불러옵니다.
    loaded_df = load_raw_csv(csv_path)
    # 불러온 데이터의 앞 5줄을 출력하여 확인합니다.
    print("CSV Head:")
    print(loaded_df.head())

    # --- 3. save_processed_parquet 함수 테스트 ---
    # "Parquet 저장 시작" 메시지를 출력합니다.
    print("\n--- Saving data to Parquet format ---")
    # 불러온 DataFrame을 Parquet 형식으로 저장합니다.
    save_processed_parquet(loaded_df, parquet_path)
    # Parquet 파일이 저장된 경로를 메시지로 출력합니다.
    print(f"Data saved to Parquet format at {parquet_path}")

    # --- 4. Parquet 파일 로드 테스트 (참고) ---
    # "Parquet 파일 확인 시작" 메시지를 출력합니다.
    print("\n--- Verifying the Parquet file ---")
    # pandas로 Parquet 파일을 다시 읽어서 제대로 저장되었는지 확인합니다.
    parquet_df = pd.read_parquet(parquet_path)
    # 읽어온 데이터의 앞 5줄을 출력하여 확인합니다.
    print("Parquet Head:")
    print(parquet_df.head())