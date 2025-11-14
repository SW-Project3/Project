"""
main/prophet_strategy_connector.py

Prophet 모델과 전략 신호를 연결하는 모듈
- Prophet 모델 예측 결과를 전략 신호 생성에 사용할 수 있는 형태로 변환
- 롤링 윈도우 방식으로 각 시점마다 예측 수행
- 예측 결과를 DataFrame으로 변환하여 strategy.generate_signals()와 연결
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Union
from pathlib import Path
import sys

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from model.prophet_model import ProphetModel
except ImportError as e:
    ProphetModel = None
    print(f"[prophet_strategy_connector] ProphetModel import 실패: {e}")

from main.strategy import generate_signals, compute_indicators, align_prophet_forecast, DEFAULT_PARAMS


def prophet_dict_to_dataframe(
    prophet_result: Dict,
    timestamp: pd.Timestamp
) -> pd.DataFrame:
    """
    Prophet 모델의 딕셔너리 형태 예측 결과를 DataFrame으로 변환
    
    매개변수:
        - prophet_result: {"yhat": float, "yhat_lower": float, "yhat_upper": float}
        - timestamp: 예측 시점
    
    반환값:
        - DataFrame with columns: ds, yhat, yhat_lower, yhat_upper
    """
    if prophet_result is None:
        return pd.DataFrame(columns=["ds", "yhat", "yhat_lower", "yhat_upper"])
    
    return pd.DataFrame({
        "ds": [timestamp],
        "yhat": [prophet_result.get("yhat", np.nan)],
        "yhat_lower": [prophet_result.get("yhat_lower", np.nan)],
        "yhat_upper": [prophet_result.get("yhat_upper", np.nan)]
    })


def generate_prophet_forecast_rolling(
    df: pd.DataFrame,
    prophet_model: ProphetModel,
    window_size: Optional[int] = None,
    step_size: int = 1,
    min_window: int = 200
) -> pd.DataFrame:
    """
    롤링 윈도우 방식으로 각 시점마다 Prophet 예측을 수행
    
    매개변수:
        - df: 가격 데이터 (Close, Volume 컬럼 필요, 소문자 또는 대문자)
        - prophet_model: ProphetModel 인스턴스
        - window_size: 롤링 윈도우 크기 (None이면 전체 데이터 사용)
        - step_size: 예측 간격 (1이면 매 시점마다, 10이면 10시점마다)
        - min_window: 최소 윈도우 크기 (Prophet 학습에 필요한 최소 데이터)
    
    반환값:
        - DataFrame with columns: ds, yhat, yhat_lower, yhat_upper
    """
    if ProphetModel is None:
        raise ImportError("ProphetModel을 import할 수 없습니다.")
    
    df = df.copy()
    
    # 컬럼명 정규화 (소문자로 통일)
    column_mapping = {
        'Open': 'open', 'High': 'high', 'Low': 'low', 
        'Close': 'close', 'Volume': 'volume'
    }
    for upper, lower in column_mapping.items():
        if upper in df.columns and lower not in df.columns:
            df[lower] = df[upper]
    
    # 필수 컬럼 확인
    if 'close' not in df.columns or 'volume' not in df.columns:
        raise ValueError("DataFrame에 'close'와 'volume' 컬럼이 필요합니다.")
    
    # 인덱스가 datetime인지 확인
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError("DataFrame의 인덱스가 datetime이어야 합니다.")
    
    # UTC 시간대로 통일
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')
    
    # 정렬
    df = df.sort_index()
    
    # 윈도우 크기 결정
    if window_size is None:
        window_size = len(df)
    
    window_size = max(window_size, min_window)
    
    # 예측 결과 저장 리스트
    forecast_list = []
    
    # 롤링 윈도우로 예측 수행
    for i in range(window_size, len(df), step_size):
        # 윈도우 데이터 추출
        window_df = df.iloc[i - window_size:i].copy()
        
        # 현재 시점
        current_timestamp = df.index[i]
        
        # Prophet 예측 수행
        try:
            prophet_result = prophet_model.fit_predict(window_df[['close', 'volume']])
            
            if prophet_result:
                # DataFrame으로 변환
                forecast_df = prophet_dict_to_dataframe(prophet_result, current_timestamp)
                forecast_list.append(forecast_df)
        except Exception as e:
            print(f"[generate_prophet_forecast_rolling] 시점 {current_timestamp} 예측 실패: {e}")
            continue
    
    # 결과 병합
    if forecast_list:
        result_df = pd.concat(forecast_list, ignore_index=True)
        result_df['ds'] = pd.to_datetime(result_df['ds'])
        return result_df
    else:
        return pd.DataFrame(columns=["ds", "yhat", "yhat_lower", "yhat_upper"])


def generate_signals_with_prophet(
    df: pd.DataFrame,
    prophet_model: Optional[ProphetModel] = None,
    prophet_forecast: Optional[pd.DataFrame] = None,
    params: Optional[Dict] = None,
    use_rolling_forecast: bool = True,
    rolling_window: Optional[int] = None,
    rolling_step: int = 1
) -> pd.DataFrame:
    """
    Prophet 예측과 전략 신호를 연결하여 신호 생성
    
    매개변수:
        - df: 가격 데이터 (지표 계산 가능한 형태)
        - prophet_model: ProphetModel 인스턴스 (use_rolling_forecast=True일 때 필요)
        - prophet_forecast: 이미 계산된 Prophet 예측 DataFrame (우선 사용)
        - params: 전략 파라미터
        - use_rolling_forecast: 롤링 윈도우 예측 사용 여부
        - rolling_window: 롤링 윈도우 크기
        - rolling_step: 롤링 윈도우 스텝 크기
    
    반환값:
        - 신호 DataFrame (signal, reason, entry_price, stop_loss, take_profit)
    """
    if params is None:
        params = DEFAULT_PARAMS
    
    df = df.copy()
    
    # 지표 계산 (없는 경우)
    needed = {"RSI", "MACD_Hist", "BB_MA", "ATR"}
    if not needed.issubset(df.columns):
        df = compute_indicators(df, params)
    
    # Prophet 예측 결과 준비
    if prophet_forecast is not None:
        # 이미 계산된 예측 결과 사용
        forecast_df = prophet_forecast.copy()
    elif prophet_model is not None and use_rolling_forecast:
        # 롤링 윈도우로 예측 수행
        forecast_df = generate_prophet_forecast_rolling(
            df, 
            prophet_model, 
            window_size=rolling_window,
            step_size=rolling_step
        )
    else:
        raise ValueError(
            "prophet_forecast 또는 prophet_model이 제공되어야 합니다. "
            "또는 use_rolling_forecast=False로 설정하고 prophet_forecast를 제공하세요."
        )
    
    # 신호 생성
    signals = generate_signals(df, forecast_df, params)
    
    return signals


def create_prophet_model_from_config(
    freq: str = "1d",
    horizon: int = 3,
    start_date: str = "2025-01-01",
    end_date: Optional[str] = None,
    **kwargs
) -> Optional[ProphetModel]:
    """
    설정으로부터 ProphetModel 인스턴스 생성
    
    매개변수:
        - freq: 예측 주기
        - horizon: 예측 스텝 수
        - start_date: 학습 시작일
        - end_date: 학습 종료일
        - **kwargs: ProphetModel의 추가 파라미터
    
    반환값:
        - ProphetModel 인스턴스
    """
    if ProphetModel is None:
        print("[create_prophet_model_from_config] ProphetModel을 사용할 수 없습니다.")
        return None
    
    return ProphetModel(
        freq=freq,
        horizon=horizon,
        start_date=start_date,
        end_date=end_date,
        **kwargs
    )


def run_full_pipeline(
    df: pd.DataFrame,
    prophet_model: Optional[ProphetModel] = None,
    strategy_params: Optional[Dict] = None,
    use_rolling_forecast: bool = True,
    rolling_window: Optional[int] = None
) -> Dict:
    """
    전체 파이프라인 실행: 데이터 → 지표 계산 → Prophet 예측 → 신호 생성
    
    매개변수:
        - df: 가격 데이터
        - prophet_model: ProphetModel 인스턴스
        - strategy_params: 전략 파라미터
        - use_rolling_forecast: 롤링 윈도우 예측 사용 여부
        - rolling_window: 롤링 윈도우 크기
    
    반환값:
        - {
            "df_with_indicators": DataFrame,
            "prophet_forecast": DataFrame,
            "signals": DataFrame
          }
    """
    df = df.copy()
    
    # 1. 지표 계산
    df_with_indicators = compute_indicators(df, strategy_params)
    
    # 2. Prophet 예측
    if prophet_model is None:
        prophet_model = create_prophet_model_from_config()
    
    if prophet_model is None:
        raise ValueError("ProphetModel을 생성할 수 없습니다.")
    
    if use_rolling_forecast:
        prophet_forecast = generate_prophet_forecast_rolling(
            df_with_indicators,
            prophet_model,
            window_size=rolling_window
        )
    else:
        # 단일 예측 (마지막 시점 기준)
        last_data = df_with_indicators[['Close', 'Volume']].copy()
        last_data.columns = ['close', 'volume']
        prophet_result = prophet_model.fit_predict(last_data)
        if prophet_result:
            last_timestamp = df_with_indicators.index[-1]
            prophet_forecast = prophet_dict_to_dataframe(prophet_result, last_timestamp)
        else:
            prophet_forecast = pd.DataFrame(columns=["ds", "yhat", "yhat_lower", "yhat_upper"])
    
    # 3. 신호 생성
    signals = generate_signals(
        df_with_indicators,
        prophet_forecast,
        strategy_params
    )
    
    return {
        "df_with_indicators": df_with_indicators,
        "prophet_forecast": prophet_forecast,
        "signals": signals
    }


if __name__ == "__main__":
    # 사용 예시
    print("=== Prophet 모델과 전략 신호 연결 테스트 ===")
    
    # 더미 데이터 생성
    import numpy as np
    dates = pd.date_range("2025-01-01", periods=300, freq="H")
    prices = 50000 + np.cumsum(np.random.randn(300) * 100)
    volumes = np.random.rand(300) * 1000
    
    demo_df = pd.DataFrame({
        "Open": prices - 10,
        "High": prices + 50,
        "Low": prices - 50,
        "Close": prices,
        "Volume": volumes
    }, index=dates)
    
    print(f"\n원본 데이터: {len(demo_df)} 행")
    print(demo_df.head())
    
    # Prophet 모델 생성
    if ProphetModel is not None:
        prophet_model = create_prophet_model_from_config(
            freq="1H",
            horizon=3,
            start_date="2025-01-01"
        )
        
        if prophet_model:
            # 전체 파이프라인 실행
            result = run_full_pipeline(
                demo_df,
                prophet_model=prophet_model,
                use_rolling_forecast=True,
                rolling_window=200
            )
            
            print(f"\n지표 계산 완료: {len(result['df_with_indicators'])} 행")
            print(f"Prophet 예측: {len(result['prophet_forecast'])} 행")
            print(f"생성된 신호: {len(result['signals'][result['signals']['signal'] != 0])} 개")
            
            # 신호 확인
            signals_with_action = result['signals'][result['signals']['signal'] != 0]
            if len(signals_with_action) > 0:
                print("\n생성된 신호:")
                print(signals_with_action.head())
        else:
            print("ProphetModel을 생성할 수 없습니다.")
    else:
        print("ProphetModel을 import할 수 없습니다. prophet 패키지가 설치되어 있는지 확인하세요.")

