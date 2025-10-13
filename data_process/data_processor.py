# data_processor.py

# --- 라이브러리 불러오기 ---
# 'pandas'는 데이터를 표(DataFrame) 형태로 다루는 라이브러리입니다.
import pandas as pd
# 'numpy'는 수치 계산, 특히 배열 연산을 위한 강력한 라이브러리입니다.
import numpy as np
# 'typing'은 코드의 변수나 함수의 타입(형식)을 명시하여 가독성과 안정성을 높여주는 도구입니다.
from typing import Tuple
# 'logging'은 프로그램의 실행 상태나 오류를 기록(로그)으로 남기기 위한 라이브러리입니다.
import logging

# --- 기본 설정 ---
# 로그를 어떤 형식으로, 어떤 수준(INFO 이상)까지 출력할지 기본 설정을 합니다.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. 기본 데이터 정리 함수 ---
# basic_clean 함수를 정의합니다. DataFrame을 받아 기본적인 정제 작업을 수행합니다.
def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame의 기본적인 정리 작업을 수행합니다.
    (컬럼 소문자화, UTC 변환, 정렬, 중복 제거, 숫자형 변환, 유효하지 않은 데이터 제거, 결측치 제거)
    """
    # "기본 클리닝 작업을 시작합니다" 라는 로그를 남깁니다.
    logging.info("Starting basic data cleaning.")

    # 만약 입력받은 DataFrame이 비어있다면,
    if df.empty:
        # 경고 로그를 남기고 그대로 반환합니다.
        logging.warning("Input DataFrame is empty. Returning as is.")
        return df

    # --- 데이터 클리닝 작업 수행 ---
    # DataFrame의 복사본을 만들어 원본 데이터가 변경되지 않도록 합니다.
    cleaned_df = df.copy()

    # 1. 컬럼 이름을 모두 소문자로 변경합니다.
    cleaned_df.columns = [col.lower() for col in cleaned_df.columns]

    # 2. 필수 컬럼이 모두 존재하는지 확인합니다.
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in cleaned_df.columns for col in cleaned_df.columns):
        # 필수 컬럼이 하나라도 없으면 에러를 발생시킵니다.
        raise ValueError(f"Input DataFrame must contain all required columns: {required_columns}")

    # 3. 인덱스(시간)가 UTC 시간대가 아니면 변환합니다.
    if cleaned_df.index.tz is None:
        # 인덱스에 UTC 시간대 정보를 부여합니다.
        cleaned_df = cleaned_df.tz_localize('UTC')
    else:
        # 이미 다른 시간대 정보가 있다면 UTC 시간대로 변환합니다.
        cleaned_df = cleaned_df.tz_convert('UTC')

    # 4. 시간순으로 정렬하고, 같은 시간(timestamp)의 중복 데이터는 마지막 것만 남깁니다.
    cleaned_df.sort_index(inplace=True)
    cleaned_df = cleaned_df[~cleaned_df.index.duplicated(keep='last')]

    # 5. 모든 컬럼을 숫자형(float)으로 변환합니다. 변환할 수 없는 값은 NaN(결측치)으로 처리합니다.
    for col in required_columns:
        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')

    # 6. 가격(OHLC) 데이터 중 0 또는 음수인 값, 거래량(volume)이 음수인 값을 가진 행을 제거합니다.
    # 이런 데이터는 비정상적인 데이터로 간주합니다.
    original_rows = len(cleaned_df)
    cleaned_df = cleaned_df[
        (cleaned_df['open'] > 0) & (cleaned_df['high'] > 0) &
        (cleaned_df['low'] > 0) & (cleaned_df['close'] > 0) &
        (cleaned_df['volume'] >= 0)
        ]
    # 얼마나 많은 행이 제거되었는지 로그를 남깁니다.
    logging.info(f"Removed {original_rows - len(cleaned_df)} rows with non-positive price/volume.")

    # 7. NaN(결측치)이 포함된 행을 모두 제거합니다.
    original_rows = len(cleaned_df)
    cleaned_df.dropna(inplace=True)
    # 얼마나 많은 행이 제거되었는지 로그를 남깁니다.
    logging.info(f"Removed {original_rows - len(cleaned_df)} rows with NaN values.")

    # 만약 모든 데이터가 제거되어 DataFrame이 비어있다면,
    if cleaned_df.empty:
        # 에러를 발생시킵니다.
        raise ValueError("DataFrame became empty after cleaning. Check raw data quality.")

    # "클리닝 작업 완료" 로그를 남깁니다.
    logging.info("Basic data cleaning finished.")
    # 최종적으로 정리된 DataFrame을 반환합니다.
    return cleaned_df

# --- 2. 논리적 오류 필터링 함수 ---
# logical_filter 함수를 정의합니다. 캔들 데이터의 논리적 오류를 검사하고 제거합니다.
def logical_filter(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    가격의 논리적 오류 (low <= open/close <= high)를 위배하는 캔들을 제거합니다.
    """
    # "논리적 필터링을 시작합니다" 라는 로그를 남깁니다.
    logging.info("Applying logical filter (low <= open/close <= high).")

    # 필터링 전의 행 개수를 저장합니다.
    original_rows = len(df)

    # low <= open, low <= close, high >= open, high >= close 조건을 모두 만족하는 행만 남깁니다.
    filtered_df = df[
        (df['low'] <= df['open']) & (df['low'] <= df['close']) &
        (df['high'] >= df['open']) & (df['high'] >= df['close'])
        ].copy() # .copy()를 사용하여 경고 메시지를 방지합니다.

    # 제거된 행의 개수를 계산합니다.
    removed_count = original_rows - len(filtered_df)
    # "몇 개의 논리적 오류 행을 제거했습니다" 라는 로그를 남깁니다.
    logging.info(f"Removed {removed_count} rows that violated logical constraints.")

    # 필터링된 DataFrame과 제거된 개수를 함께 반환합니다.
    return filtered_df, removed_count

# --- 3. 가격 급등(Spike) 필터링 함수 ---
# spike_filter 함수를 정의합니다. 비정상적인 가격 급등/급락 캔들을 제거합니다.
def spike_filter(df: pd.DataFrame, threshold: float = 0.20) -> Tuple[pd.DataFrame, int]:
    """
    이전 종가 대비 절대 수익률이 threshold를 초과하는 비정상적인 캔들을 제거합니다.
    """
    # "가격 급등 필터링을 시작합니다" 라는 로그를 남깁니다.
    logging.info(f"Applying spike filter with threshold {threshold:.2%}.")

    # 필터링 전의 행 개수를 저장합니다.
    original_rows = len(df)

    # 이전 종가 대비 현재 종가의 수익률을 계산합니다. pct_change() 함수를 사용합니다.
    # abs()로 절대값을 취해 급등과 급락을 모두 확인합니다.
    returns = df['close'].pct_change().abs()

    # 수익률이 지정된 임계값(threshold) 이하인 정상적인 캔들만 남깁니다.
    filtered_df = df[returns <= threshold].copy()

    # 제거된 행의 개수를 계산합니다.
    removed_count = original_rows - len(filtered_df)

    # 어떤 캔들이 제거되었는지 로그로 남기기 위해, 제거된 캔들의 인덱스(시간)를 찾습니다.
    removed_timestamps = df.index.difference(filtered_df.index)
    # 만약 제거된 캔들이 있다면,
    if removed_count > 0:
        # "몇 개의 가격 급등 캔들을 제거했습니다" 라는 경고 로그를 남깁니다.
        logging.warning(f"Removed {removed_count} spike candles. Timestamps: {removed_timestamps.to_list()}")
    # 제거된 캔들이 없다면,
    else:
        # "가격 급등 캔들을 찾지 못했습니다" 라는 로그를 남깁니다.
        logging.info("No spike candles found.")

    # 필터링된 DataFrame과 제거된 개수를 함께 반환합니다.
    return filtered_df, removed_count

# --- 4. 데이터 연속성 보장 함수 ---
# ensure_continuity 함수를 정의합니다. 데이터의 시간 간격이 일정한지 확인하고 처리합니다.
def ensure_continuity(df: pd.DataFrame, freq: str, how: str = "drop") -> Tuple[pd.DataFrame, int]:
    """
    DataFrame의 시작부터 끝까지 지정된 시간 간격(freq)으로 인덱스를 재생성하고,
    누락된 바(행)가 있다면 개수를 계산한 후 drop 또는 ffill 방식으로 처리합니다.
    """
    # "데이터 연속성 검사를 시작합니다" 라는 로그를 남깁니다.
    logging.info(f"Ensuring data continuity with frequency '{freq}' (method: {how}).")

    # DataFrame의 시작 시간과 끝 시간을 기준으로, 일정한 시간 간격(freq)을 갖는 새로운 인덱스를 생성합니다.
    continuous_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq, tz='UTC')

    # 기존 데이터에 새로운 인덱스를 적용합니다. 이 과정에서 비어있는 시간은 NaN 값으로 채워집니다.
    continuous_df = df.reindex(continuous_index)

    # 비어있는 행(결측치)의 개수를 계산합니다.
    missing_count = continuous_df['close'].isna().sum()
    # "몇 개의 누락된 바를 발견했습니다" 라는 로그를 남깁니다.
    logging.info(f"Found {missing_count} missing bars.")

    # 처리 방식('how')에 따라 누락된 데이터를 처리합니다.
    if how == "drop":
        # 'drop' 방식이면, 비어있는 모든 행을 제거합니다.
        result_df = continuous_df.dropna()
        # "누락된 바를 모두 제거했습니다" 라는 로그를 남깁니다.
        logging.info("Dropped all missing bars.")
    elif how == "ffill":
        # 'ffill' 방식이면, 비어있는 값을 바로 이전 행의 값으로 채웁니다. (forward-fill)
        result_df = continuous_df.ffill()
        # "누락된 바를 이전 값으로 채웠습니다" 라는 로그를 남깁니다.
        logging.info("Forward-filled all missing bars.")
    else:
        # 지원하지 않는 방식이면 에러를 발생시킵니다.
        raise ValueError("Invalid 'how' method. Choose from 'drop' or 'ffill'.")

    # 처리된 DataFrame과 누락된 개수를 함께 반환합니다.
    return result_df, missing_count

# --- 5. 리샘플링 함수 ---
# resample_ohlcv 함수를 정의합니다. 더 짧은 시간 간격의 데이터를 더 긴 시간 간격으로 변환합니다. (예: 1분봉 -> 5분봉)
def resample_ohlcv(df: pd.DataFrame, out_freq: str) -> pd.DataFrame:
    """
    OHLCV 데이터를 지정된 시간 간격(out_freq)으로 리샘플링합니다.
    """
    # "데이터를 리샘플링합니다" 라는 로그를 남깁니다.
    logging.info(f"Resampling data to '{out_freq}'.")

    # 리샘플링 규칙을 딕셔너리로 정의합니다.
    resampling_rules = {
        'open': 'first',  # 시가는 해당 기간의 첫 번째 값
        'high': 'max',    # 고가는 해당 기간의 최대값
        'low': 'min',     # 저가는 해당 기간의 최소값
        'close': 'last',  # 종가는 해당 기간의 마지막 값
        'volume': 'sum'   # 거래량은 해당 기간의 모든 값을 합산
    }

    # pandas의 resample 기능을 사용해 데이터를 지정된 규칙에 따라 변환합니다.
    # dropna()는 모든 값이 비어있는 (데이터가 전혀 없는) 그룹은 결과에서 제외합니다.
    resampled_df = df.resample(out_freq).apply(resampling_rules).dropna()

    # "리샘플링 완료" 로그를 남깁니다.
    logging.info(f"Resampling complete. Result has {len(resampled_df)} rows.")

    # 리샘플링된 DataFrame을 반환합니다.
    return resampled_df

# --- 예제 코드 실행 부분 ---

# 이 파일에 있는 모든 전처리 함수를 테스트하기 위한 예제 함수를 정의합니다.
def run_all_tests():
    """이 파일에 있는 모든 전처리 함수를 테스트하기 위한 예제 함수입니다."""
    # --- 테스트용 데이터 생성 ---
    # 테스트 시작을 알리는 메시지를 출력합니다.
    print("\n--- Creating sample data for testing ---")
    # 테스트 데이터의 시작 시간을 정의합니다.
    start_time = "2025-01-01 00:00:00"
    # '2025-01-01'부터 1분 간격으로 2000개의 가상 시간 인덱스를 생성합니다.
    index = pd.date_range(start=start_time, periods=2000, freq="1min", tz='UTC')
    # open, high, low, close, volume에 대한 무작위 숫자 데이터로 구성된 딕셔너리를 만듭니다.
    data = {
        'open': np.random.uniform(100, 102, size=2000),
        'high': np.random.uniform(102, 104, size=2000),
        'low': np.random.uniform(98, 100, size=2000),
        'close': np.random.uniform(100, 102, size=2000),
        'volume': np.random.uniform(10, 50, size=2000)
    }
    # 위에서 만든 시간 인덱스와 데이터로 테스트용 DataFrame(표)을 생성합니다.
    sample_df = pd.DataFrame(data, index=index)

    # --- 일부러 오류 데이터 주입 ---
    # 테스트를 위해 일부러 논리적 오류(고가 < 저가)를 만듭니다. (1번 행의 2번 열)
    sample_df.iloc[1, 2] = 90
    # 테스트를 위해 일부러 가격 급등(spike) 오류를 만듭니다. (2번 행의 3번 열)
    sample_df.iloc[2, 3] = 200
    # 테스트를 위해 일부러 음수 가격 오류를 만듭니다. (3번 행의 0번 열)
    sample_df.iloc[3, 0] = -100
    # 테스트를 위해 일부러 5분간의 데이터 누락(시간적 갭)을 만듭니다. (5~9번 행 삭제)
    sample_df = sample_df.drop(sample_df.index[5:10])
    # 오류 데이터가 포함된 테스트 데이터가 생성되었음을 알립니다.
    print("Sample data created with intentional errors.")

    # --- 각 함수 테스트 실행 ---
    # 1. basic_clean 함수를 테스트합니다.
    print("\n--- 1. Testing basic_clean ---")
    cleaned_df = basic_clean(sample_df)
    print(f"Original rows: {len(sample_df)}, Cleaned rows: {len(cleaned_df)}")

    # 2. logical_filter 함수를 테스트합니다.
    print("\n--- 2. Testing logical_filter ---")
    logical_df, removed_logical = logical_filter(cleaned_df)
    print(f"Rows before logical filter: {len(cleaned_df)}, After: {len(logical_df)}")

    # 3. spike_filter 함수를 테스트합니다.
    print("\n--- 3. Testing spike_filter ---")
    spike_df, removed_spike = spike_filter(logical_df)
    print(f"Rows before spike filter: {len(logical_df)}, After: {len(spike_df)}")

    # 4. ensure_continuity 함수를 테스트합니다.
    print("\n--- 4. Testing ensure_continuity ---")
    continuous_df, missing = ensure_continuity(spike_df, freq="1min")
    print(f"Found and dropped {missing} missing bars.")

    # 5. resample_ohlcv 함수를 테스트합니다.
    print("\n--- 5. Testing resample_ohlcv ---")
    resampled_5m_df = resample_ohlcv(continuous_df, out_freq="5min")
    print("Resampled to 5min frequency. Head of the result:")
    print(resampled_5m_df.head())


if __name__ == '__main__':
    # 테스트를 실행하고 싶으면 아래 줄의 '#'을 제거하고 실행
    #run_all_tests()

    # 평소에는 아무것도 실행되지 않도록 비워두거나 pass
    pass