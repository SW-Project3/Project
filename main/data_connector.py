"""
main/data_connector.py

데이터 로딩 및 지표 계산을 연결하는 모듈
- data_process의 데이터 로딩 함수와 main/indicators를 연결
- 컬럼명 변환 및 데이터 형식 통일
"""

import pandas as pd
from typing import Union, Optional, Dict
from pathlib import Path

from main import indicators
from main.strategy import compute_indicators, DEFAULT_PARAMS


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    data_process에서 로드한 데이터의 소문자 컬럼명을 
    main 모듈에서 사용하는 대문자 컬럼명으로 변환
    
    변환 규칙:
    - open -> Open
    - high -> High
    - low -> Low
    - close -> Close
    - volume -> Volume
    """
    df = df.copy()
    
    column_mapping = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }
    
    # 소문자 컬럼명을 대문자로 변환
    for lower_col, upper_col in column_mapping.items():
        if lower_col in df.columns:
            df[upper_col] = df[lower_col]
            # 원본 소문자 컬럼은 유지 (필요시 삭제 가능)
    
    return df


def load_data_with_indicators(
    data_path: Union[str, Path],
    params: Optional[Dict] = None,
    normalize_columns: bool = True
) -> pd.DataFrame:
    """
    데이터 파일을 로드하고 지표를 계산하여 반환
    
    매개변수:
        - data_path: CSV 또는 Parquet 파일 경로
        - params: 지표 계산 파라미터 (None이면 DEFAULT_PARAMS 사용)
        - normalize_columns: 컬럼명 정규화 여부
    
    반환값:
        - 지표가 계산된 DataFrame (RSI, MACD, BB, ATR 포함)
    """
    # 파일 확장자 확인
    filepath = Path(data_path)
    
    if not filepath.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {filepath}")
    
    # 파일 형식에 따라 로드
    if filepath.suffix.lower() == '.parquet':
        df = pd.read_parquet(filepath)
    elif filepath.suffix.lower() == '.csv':
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {filepath.suffix}")
    
    # 인덱스가 datetime이 아니면 변환 시도
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        else:
            raise ValueError("DataFrame의 인덱스가 datetime이 아니고 'timestamp' 컬럼도 없습니다.")
    
    # 컬럼명 정규화
    if normalize_columns:
        df = normalize_column_names(df)
    
    # 필수 컬럼 확인
    required_cols = ['Open', 'High', 'Low', 'Close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        # 소문자로도 확인
        lower_mapping = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}
        for lower_col, upper_col in lower_mapping.items():
            if lower_col in df.columns and upper_col not in df.columns:
                df[upper_col] = df[lower_col]
        
        # 다시 확인
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"필수 컬럼이 없습니다: {missing_cols}")
    
    # 지표 계산
    df_with_indicators = compute_indicators(df, params)
    
    return df_with_indicators


def load_from_data_process(
    raw_csv_path: Optional[Union[str, Path]] = None,
    processed_csv_path: Optional[Union[str, Path]] = None,
    processed_parquet_path: Optional[Union[str, Path]] = None,
    params: Optional[Dict] = None,
    use_processed: bool = True
) -> pd.DataFrame:
    """
    data_process 모듈의 함수를 사용하여 데이터를 로드하고 지표를 계산
    
    매개변수:
        - raw_csv_path: 원본 CSV 파일 경로
        - processed_csv_path: 처리된 CSV 파일 경로
        - processed_parquet_path: 처리된 Parquet 파일 경로
        - params: 지표 계산 파라미터
        - use_processed: 처리된 데이터 사용 여부 (True면 processed 우선)
    
    반환값:
        - 지표가 계산된 DataFrame
    """
    # data_process 모듈 import (상대 경로)
    try:
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        from data_process.data_loader import load_raw_csv
        from data_process.data_processor import process_raw_to_timeframe, basic_clean, logical_filter, spike_filter, resample_ohlcv
    except ImportError as e:
        raise ImportError(f"data_process 모듈을 import할 수 없습니다: {e}")
    
    # 처리된 데이터 우선 사용
    if use_processed:
        if processed_parquet_path and Path(processed_parquet_path).exists():
            return load_data_with_indicators(processed_parquet_path, params)
        elif processed_csv_path and Path(processed_csv_path).exists():
            return load_data_with_indicators(processed_csv_path, params)
        elif raw_csv_path and Path(raw_csv_path).exists():
            # 원본 데이터를 로드하고 처리
            df = load_raw_csv(raw_csv_path)
            df = normalize_column_names(df)
            df_with_indicators = compute_indicators(df, params)
            return df_with_indicators
        else:
            raise FileNotFoundError("사용 가능한 데이터 파일을 찾을 수 없습니다.")
    else:
        # 원본 데이터만 사용
        if raw_csv_path and Path(raw_csv_path).exists():
            df = load_raw_csv(raw_csv_path)
            df = normalize_column_names(df)
            df_with_indicators = compute_indicators(df, params)
            return df_with_indicators
        else:
            raise FileNotFoundError(f"원본 CSV 파일을 찾을 수 없습니다: {raw_csv_path}")


def add_indicators_to_dataframe(
    df: pd.DataFrame,
    params: Optional[Dict] = None,
    normalize_columns: bool = True
) -> pd.DataFrame:
    """
    기존 DataFrame에 지표를 추가
    
    매개변수:
        - df: 가격 데이터 DataFrame
        - params: 지표 계산 파라미터
        - normalize_columns: 컬럼명 정규화 여부
    
    반환값:
        - 지표가 추가된 DataFrame
    """
    df = df.copy()
    
    # 컬럼명 정규화
    if normalize_columns:
        df = normalize_column_names(df)
    
    # 지표 계산
    df_with_indicators = compute_indicators(df, params)
    
    return df_with_indicators


def get_data_summary(df: pd.DataFrame) -> Dict:
    """
    데이터와 지표의 요약 정보 반환
    
    반환값:
        - 데이터 행 수, 기간, 컬럼 목록, 지표 존재 여부 등
    """
    summary = {
        "rows": len(df),
        "columns": list(df.columns),
        "date_range": None,
        "has_indicators": False,
        "indicator_list": []
    }
    
    # 날짜 범위
    if pd.api.types.is_datetime64_any_dtype(df.index):
        summary["date_range"] = {
            "start": df.index.min(),
            "end": df.index.max()
        }
    
    # 지표 확인
    indicator_columns = ['RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 
                        'BB_MA', 'BB_Upper', 'BB_Lower', 'BB_Width', 'ATR']
    existing_indicators = [col for col in indicator_columns if col in df.columns]
    
    if existing_indicators:
        summary["has_indicators"] = True
        summary["indicator_list"] = existing_indicators
    
    return summary


if __name__ == "__main__":
    # 사용 예시
    print("=== 데이터와 지표 연결 테스트 ===")
    
    # 예시 1: 직접 파일 경로로 로드
    # df = load_data_with_indicators("data/processed/binance_BTCUSDT_3m_processed.parquet")
    # print(f"로드된 데이터 행 수: {len(df)}")
    # print(f"컬럼: {df.columns.tolist()}")
    
    # 예시 2: 기존 DataFrame에 지표 추가
    import numpy as np
    dates = pd.date_range("2025-01-01", periods=100, freq="H")
    prices = 50000 + np.cumsum(np.random.randn(100) * 100)
    
    demo_df = pd.DataFrame({
        "open": prices - 10,
        "high": prices + 50,
        "low": prices - 50,
        "close": prices,
        "volume": np.random.rand(100) * 1000
    }, index=dates)
    
    print("\n원본 데이터:")
    print(demo_df.head())
    print(f"컬럼: {demo_df.columns.tolist()}")
    
    # 지표 추가
    df_with_indicators = add_indicators_to_dataframe(demo_df)
    
    print("\n지표 추가 후:")
    print(df_with_indicators.head())
    print(f"컬럼: {df_with_indicators.columns.tolist()}")
    
    # 요약 정보
    summary = get_data_summary(df_with_indicators)
    print("\n데이터 요약:")
    print(f"행 수: {summary['rows']}")
    print(f"날짜 범위: {summary['date_range']}")
    print(f"지표 존재: {summary['has_indicators']}")
    print(f"지표 목록: {summary['indicator_list']}")

