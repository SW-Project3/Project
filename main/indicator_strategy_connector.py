"""
main/indicator_strategy_connector.py

지표 계산과 전략 신호 생성을 연결하는 모듈
- 지표를 계산하고 바로 전략 신호로 변환
- 지표 기반 신호 생성 (Prophet 없이도 가능)
- 지표 값 분석 및 신호 필터링
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple

from main import indicators
from main.strategy import compute_indicators, generate_signals, DEFAULT_PARAMS


def calculate_all_indicators(
    df: pd.DataFrame,
    params: Optional[Dict] = None
) -> pd.DataFrame:
    """
    모든 지표를 계산하여 DataFrame에 추가
    
    매개변수:
        - df: 가격 데이터 (Open, High, Low, Close, Volume)
        - params: 지표 계산 파라미터
    
    반환값:
        - 지표가 추가된 DataFrame
    """
    return compute_indicators(df, params)


def generate_signals_from_indicators(
    df: pd.DataFrame,
    params: Optional[Dict] = None,
    use_prophet: bool = False,
    prophet_forecast: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    지표를 기반으로 전략 신호 생성
    
    매개변수:
        - df: 가격 데이터 (지표 포함 또는 미포함)
        - params: 전략 파라미터
        - use_prophet: Prophet 예측 사용 여부
        - prophet_forecast: Prophet 예측 결과 (use_prophet=True일 때 필요)
    
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
    
    # Prophet 사용 여부에 따라 신호 생성
    if use_prophet and prophet_forecast is not None:
        signals = generate_signals(df, prophet_forecast, params)
    else:
        # Prophet 없이 지표만으로 신호 생성
        signals = generate_signals_indicator_only(df, params)
    
    return signals


def generate_signals_indicator_only(
    df: pd.DataFrame,
    params: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Prophet 없이 지표만으로 신호 생성
    
    매개변수:
        - df: 지표가 포함된 가격 데이터
        - params: 전략 파라미터
    
    반환값:
        - 신호 DataFrame
    """
    if params is None:
        params = DEFAULT_PARAMS
    
    df = df.copy()
    
    # 지표 계산 보증
    needed = {"RSI", "MACD_Hist", "BB_MA", "ATR", "Close"}
    if not needed.issubset(df.columns):
        df = compute_indicators(df, params)
    
    # 신호 DataFrame 초기화
    signals = pd.DataFrame(index=df.index)
    signals["signal"] = 0
    signals["reason"] = ""
    signals["entry_price"] = np.nan
    signals["stop_loss"] = np.nan
    signals["take_profit"] = np.nan
    
    # 파라미터
    rsi_buy = params["rsi_buy"]
    rsi_sell = params["rsi_sell"]
    sl_mult = params["stop_loss_atr_mult"]
    tp_mult = params["take_profit_atr_mult"]
    
    holding = 0
    
    for ts in df.index:
        price = df.at[ts, "Close"]
        rsi = df.at[ts, "RSI"]
        macd_hist = df.at[ts, "MACD_Hist"]
        atr = df.at[ts, "ATR"]
        bb_upper = df.at[ts, "BB_Upper"]
        bb_lower = df.at[ts, "BB_Lower"]
        bb_ma = df.at[ts, "BB_MA"]
        
        # 필요한 값 체크
        if pd.isna(price) or pd.isna(rsi) or pd.isna(macd_hist) or pd.isna(atr):
            continue
        
        # 지표 기반 매수/매도 조건
        # 매수 조건: RSI 과매도 + MACD 상승 + 가격이 볼린저 하단 근처
        buy_cond = (
            (rsi <= rsi_buy) and 
            (macd_hist > 0) and 
            (price <= bb_lower * 1.02)  # 볼린저 하단 근처
        )
        
        # 매도 조건: RSI 과매수 + MACD 하락 + 가격이 볼린저 상단 근처
        sell_cond = (
            (rsi >= rsi_sell) and 
            (macd_hist < 0) and 
            (price >= bb_upper * 0.98)  # 볼린저 상단 근처
        )
        
        if buy_cond and holding <= 0:
            signals.at[ts, "signal"] = 1
            signals.at[ts, "reason"] = (
                f"RSI={rsi:.1f} MACD={macd_hist:.6f} "
                f"BB_Lower={bb_lower:.2f} Price={price:.2f}"
            )
            signals.at[ts, "entry_price"] = price
            signals.at[ts, "stop_loss"] = price - sl_mult * atr
            signals.at[ts, "take_profit"] = price + tp_mult * atr
            holding = 1
        
        elif sell_cond and holding >= 0:
            signals.at[ts, "signal"] = -1
            signals.at[ts, "reason"] = (
                f"RSI={rsi:.1f} MACD={macd_hist:.6f} "
                f"BB_Upper={bb_upper:.2f} Price={price:.2f}"
            )
            signals.at[ts, "entry_price"] = price
            signals.at[ts, "stop_loss"] = price + sl_mult * atr
            signals.at[ts, "take_profit"] = price - tp_mult * atr
            holding = -1
    
    return signals


def generate_signals_prophet_strategy(
    df: pd.DataFrame,
    prophet_forecast: pd.DataFrame,
    params: Optional[Dict] = None
) -> pd.DataFrame:
    """
    새로운 Prophet 기반 전략 신호 생성
    
    진입 조건:
    - yhat > close (예측 상승)
    - MACD > MACD_signal (추세 상승)
    - RSI ≤ 60 (과매수 아님)
    - close ≥ 볼린저 중단선
    
    EXIT 조건:
    - yhat ≤ close (예측 하락)
    - 또는 MACD ≤ MACD_signal
    - 또는 RSI > 70 (과매수)
    
    손절/익절: entry - 1.5*ATR / entry + 3*ATR
    """
    if params is None:
        params = DEFAULT_PARAMS
    
    df = df.copy()
    
    # 지표 계산 보증
    needed = {"RSI", "MACD", "MACD_Signal", "BB_MA", "ATR", "Close"}
    if not needed.issubset(df.columns):
        df = compute_indicators(df, params)
    
    # Prophet 예측을 가격 인덱스에 맞춰 정렬/병합
    from main.strategy import align_prophet_forecast
    pf_aligned = align_prophet_forecast(df, prophet_forecast, method="forward")
    df = df.join(pf_aligned)
    
    # Prophet 예측이 있는 시점 수 확인
    yhat_available = df["yhat"].notna().sum()
    print(f"[generate_signals_prophet_strategy] Prophet 예측 사용 가능한 시점: {yhat_available} / {len(df)}")
    
    # 신호 DataFrame 초기화
    signals = pd.DataFrame(index=df.index)
    signals["signal"] = 0
    signals["reason"] = ""
    signals["entry_price"] = np.nan
    signals["stop_loss"] = np.nan
    signals["take_profit"] = np.nan
    
    # 파라미터
    sl_mult = 1.5  # 손절: ATR × 1.5
    tp_mult = 3.0  # 익절: ATR × 3.0
    rsi_entry_max = 60  # 진입 최대 RSI
    rsi_exit_threshold = 70  # 청산 RSI
    
    position = 0  # 0: 없음, 1: 롱 포지션
    entry_price = None
    entry_atr = None
    
    for ts in df.index:
        price = df.at[ts, "Close"]
        yhat = df.at[ts, "yhat"]
        rsi = df.at[ts, "RSI"]
        macd = df.at[ts, "MACD"]
        macd_signal = df.at[ts, "MACD_Signal"]
        bb_ma = df.at[ts, "BB_MA"]
        atr = df.at[ts, "ATR"]
        
        # 필요한 값 체크
        if pd.isna(price) or pd.isna(rsi) or pd.isna(macd) or pd.isna(macd_signal) or pd.isna(atr) or pd.isna(bb_ma):
            continue
        
        if pd.isna(yhat):
            # Prophet 예측이 없으면 스킵
            continue
        
        # 포지션이 있을 때 EXIT 조건 확인
        if position == 1:
            exit_cond1 = yhat <= price  # Prophet 예측 하락
            exit_cond2 = macd <= macd_signal  # MACD 하락
            exit_cond3 = rsi > rsi_exit_threshold  # 과매수
            
            if exit_cond1 or exit_cond2 or exit_cond3:
                # 청산 신호
                signals.at[ts, "signal"] = -1
                exit_reasons = []
                if exit_cond1:
                    exit_reasons.append(f"Prophet하락(yhat={yhat:.2f}≤{price:.2f})")
                if exit_cond2:
                    exit_reasons.append(f"MACD하락({macd:.2f}≤{macd_signal:.2f})")
                if exit_cond3:
                    exit_reasons.append(f"RSI과매수({rsi:.1f}>{rsi_exit_threshold})")
                signals.at[ts, "reason"] = " | ".join(exit_reasons)
                signals.at[ts, "entry_price"] = entry_price
                signals.at[ts, "stop_loss"] = entry_price - sl_mult * entry_atr
                signals.at[ts, "take_profit"] = entry_price + tp_mult * entry_atr
                position = 0
                entry_price = None
                entry_atr = None
                continue
        
        # 포지션이 없을 때 진입 조건 확인
        if position == 0:
            cond1 = yhat > price  # Prophet 예측 상승
            cond2 = macd > macd_signal  # MACD 상승
            cond3 = rsi <= rsi_entry_max  # 과매수 아님
            cond4 = price >= bb_ma  # 볼린저 중단선 이상
            
            if cond1 and cond2 and cond3 and cond4:
                # 진입 신호
                signals.at[ts, "signal"] = 1
                signals.at[ts, "reason"] = (
                    f"Prophet상승(yhat={yhat:.2f}>{price:.2f}) "
                    f"MACD상승({macd:.2f}>{macd_signal:.2f}) "
                    f"RSI={rsi:.1f}≤{rsi_entry_max} "
                    f"BB중단선={bb_ma:.2f}≤{price:.2f}"
                )
                signals.at[ts, "entry_price"] = price
                signals.at[ts, "stop_loss"] = price - sl_mult * atr
                signals.at[ts, "take_profit"] = price + tp_mult * atr
                position = 1
                entry_price = price
                entry_atr = atr
    
    # 최종 신호 통계
    buy_signals_count = len(signals[signals["signal"] == 1])
    sell_signals_count = len(signals[signals["signal"] == -1])
    print(f"[generate_signals_prophet_strategy] 최종 신호 생성: 매수 {buy_signals_count}개, 매도 {sell_signals_count}개")
    
    return signals


def analyze_indicators(
    df: pd.DataFrame,
    params: Optional[Dict] = None
) -> Dict:
    """
    지표 값을 분석하여 현재 상태 요약
    
    매개변수:
        - df: 지표가 포함된 가격 데이터
        - params: 전략 파라미터
    
    반환값:
        - 지표 분석 결과 딕셔너리
    """
    if params is None:
        params = DEFAULT_PARAMS
    
    df = df.copy()
    
    # 지표 계산 (없는 경우)
    needed = {"RSI", "MACD_Hist", "BB_MA", "ATR", "Close"}
    if not needed.issubset(df.columns):
        df = compute_indicators(df, params)
    
    # 마지막 시점의 지표 값
    last_idx = df.index[-1]
    
    current_price = df.at[last_idx, "Close"]
    rsi = df.at[last_idx, "RSI"]
    macd_hist = df.at[last_idx, "MACD_Hist"]
    atr = df.at[last_idx, "ATR"]
    bb_upper = df.at[last_idx, "BB_Upper"]
    bb_lower = df.at[last_idx, "BB_Lower"]
    bb_ma = df.at[last_idx, "BB_MA"]
    
    # RSI 상태
    rsi_status = "과매수" if rsi >= params["rsi_sell"] else "과매도" if rsi <= params["rsi_buy"] else "중립"
    
    # MACD 상태
    macd_status = "상승" if macd_hist > 0 else "하락"
    
    # 볼린저 밴드 위치
    if current_price >= bb_upper:
        bb_position = "상단 돌파"
    elif current_price <= bb_lower:
        bb_position = "하단 돌파"
    elif current_price > bb_ma:
        bb_position = "상단 근처"
    else:
        bb_position = "하단 근처"
    
    # 종합 신호 강도 (0~100)
    signal_strength = 0
    
    # RSI 기여도
    if rsi <= params["rsi_buy"]:
        signal_strength += 30
    elif rsi >= params["rsi_sell"]:
        signal_strength -= 30
    
    # MACD 기여도
    if macd_hist > 0:
        signal_strength += 20
    else:
        signal_strength -= 20
    
    # 볼린저 밴드 기여도
    if current_price <= bb_lower:
        signal_strength += 25
    elif current_price >= bb_upper:
        signal_strength -= 25
    
    # 추천 액션
    if signal_strength >= 50:
        recommended_action = "매수 고려"
    elif signal_strength <= -50:
        recommended_action = "매도 고려"
    else:
        recommended_action = "관망"
    
    return {
        "timestamp": last_idx,
        "price": current_price,
        "rsi": rsi,
        "rsi_status": rsi_status,
        "macd_hist": macd_hist,
        "macd_status": macd_status,
        "atr": atr,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
        "bb_ma": bb_ma,
        "bb_position": bb_position,
        "signal_strength": signal_strength,
        "recommended_action": recommended_action
    }


def filter_signals_by_indicators(
    signals: pd.DataFrame,
    df: pd.DataFrame,
    min_rsi: Optional[float] = None,
    max_rsi: Optional[float] = None,
    min_macd_hist: Optional[float] = None,
    max_macd_hist: Optional[float] = None,
    min_atr: Optional[float] = None
) -> pd.DataFrame:
    """
    지표 값에 따라 신호를 필터링
    
    매개변수:
        - signals: 신호 DataFrame
        - df: 지표가 포함된 가격 데이터
        - min_rsi: 최소 RSI 값
        - max_rsi: 최대 RSI 값
        - min_macd_hist: 최소 MACD 히스토그램 값
        - max_macd_hist: 최대 MACD 히스토그램 값
        - min_atr: 최소 ATR 값
    
    반환값:
        - 필터링된 신호 DataFrame
    """
    filtered_signals = signals.copy()
    
    for idx in signals.index:
        if signals.at[idx, "signal"] == 0:
            continue
        
        # 지표 값 확인
        if idx in df.index:
            rsi = df.at[idx, "RSI"]
            macd_hist = df.at[idx, "MACD_Hist"]
            atr = df.at[idx, "ATR"]
            
            # 필터 조건 확인
            if min_rsi is not None and rsi < min_rsi:
                filtered_signals.at[idx, "signal"] = 0
                continue
            
            if max_rsi is not None and rsi > max_rsi:
                filtered_signals.at[idx, "signal"] = 0
                continue
            
            if min_macd_hist is not None and macd_hist < min_macd_hist:
                filtered_signals.at[idx, "signal"] = 0
                continue
            
            if max_macd_hist is not None and macd_hist > max_macd_hist:
                filtered_signals.at[idx, "signal"] = 0
                continue
            
            if min_atr is not None and atr < min_atr:
                filtered_signals.at[idx, "signal"] = 0
                continue
    
    return filtered_signals


def get_indicator_signals_summary(
    df: pd.DataFrame,
    signals: pd.DataFrame
) -> Dict:
    """
    지표와 신호의 요약 통계 반환
    
    매개변수:
        - df: 지표가 포함된 가격 데이터
        - signals: 신호 DataFrame
    
    반환값:
        - 요약 통계 딕셔너리
    """
    buy_signals = signals[signals["signal"] == 1]
    sell_signals = signals[signals["signal"] == -1]
    
    summary = {
        "total_signals": len(signals[signals["signal"] != 0]),
        "buy_signals": len(buy_signals),
        "sell_signals": len(sell_signals),
        "buy_signal_indicators": {},
        "sell_signal_indicators": {}
    }
    
    # 매수 신호 시점의 지표 평균
    if len(buy_signals) > 0:
        buy_indices = buy_signals.index
        buy_rsi = df.loc[buy_indices, "RSI"].mean()
        buy_macd = df.loc[buy_indices, "MACD_Hist"].mean()
        buy_atr = df.loc[buy_indices, "ATR"].mean()
        
        summary["buy_signal_indicators"] = {
            "avg_rsi": buy_rsi,
            "avg_macd_hist": buy_macd,
            "avg_atr": buy_atr
        }
    
    # 매도 신호 시점의 지표 평균
    if len(sell_signals) > 0:
        sell_indices = sell_signals.index
        sell_rsi = df.loc[sell_indices, "RSI"].mean()
        sell_macd = df.loc[sell_indices, "MACD_Hist"].mean()
        sell_atr = df.loc[sell_indices, "ATR"].mean()
        
        summary["sell_signal_indicators"] = {
            "avg_rsi": sell_rsi,
            "avg_macd_hist": sell_macd,
            "avg_atr": sell_atr
        }
    
    return summary


def run_indicator_to_signal_pipeline(
    df: pd.DataFrame,
    params: Optional[Dict] = None,
    use_prophet: bool = False,
    prophet_forecast: Optional[pd.DataFrame] = None,
    filter_signals: bool = False,
    filter_params: Optional[Dict] = None
) -> Dict:
    """
    지표 계산부터 신호 생성까지 전체 파이프라인 실행
    
    매개변수:
        - df: 가격 데이터
        - params: 전략 파라미터
        - use_prophet: Prophet 사용 여부
        - prophet_forecast: Prophet 예측 결과
        - filter_signals: 신호 필터링 여부
        - filter_params: 필터링 파라미터
    
    반환값:
        - {
            "df_with_indicators": DataFrame,
            "signals": DataFrame,
            "analysis": Dict,
            "summary": Dict
          }
    """
    df = df.copy()
    
    # 1. 지표 계산
    df_with_indicators = calculate_all_indicators(df, params)
    
    # 2. 신호 생성
    signals = generate_signals_from_indicators(
        df_with_indicators,
        params=params,
        use_prophet=use_prophet,
        prophet_forecast=prophet_forecast
    )
    
    # 3. 신호 필터링 (선택적)
    if filter_signals and filter_params:
        signals = filter_signals_by_indicators(
            signals,
            df_with_indicators,
            **filter_params
        )
    
    # 4. 지표 분석
    analysis = analyze_indicators(df_with_indicators, params)
    
    # 5. 요약 통계
    summary = get_indicator_signals_summary(df_with_indicators, signals)
    
    return {
        "df_with_indicators": df_with_indicators,
        "signals": signals,
        "analysis": analysis,
        "summary": summary
    }


if __name__ == "__main__":
    # 사용 예시
    print("=== 지표와 전략 신호 연결 테스트 ===")
    
    # 더미 데이터 생성
    dates = pd.date_range("2025-01-01", periods=200, freq="H")
    prices = 50000 + np.cumsum(np.random.randn(200) * 100)
    
    demo_df = pd.DataFrame({
        "Open": prices - 10,
        "High": prices + 50,
        "Low": prices - 50,
        "Close": prices,
        "Volume": np.random.rand(200) * 1000
    }, index=dates)
    
    print(f"\n원본 데이터: {len(demo_df)} 행")
    
    # 전체 파이프라인 실행
    result = run_indicator_to_signal_pipeline(
        demo_df,
        use_prophet=False  # Prophet 없이 지표만 사용
    )
    
    print(f"\n지표 계산 완료: {len(result['df_with_indicators'])} 행")
    print(f"생성된 신호: {result['summary']['total_signals']} 개")
    print(f"  - 매수: {result['summary']['buy_signals']} 개")
    print(f"  - 매도: {result['summary']['sell_signals']} 개")
    
    # 현재 지표 분석
    print("\n현재 지표 분석:")
    analysis = result["analysis"]
    print(f"  가격: {analysis['price']:.2f}")
    print(f"  RSI: {analysis['rsi']:.2f} ({analysis['rsi_status']})")
    print(f"  MACD: {analysis['macd_hist']:.6f} ({analysis['macd_status']})")
    print(f"  볼린저 밴드: {analysis['bb_position']}")
    print(f"  신호 강도: {analysis['signal_strength']}")
    print(f"  추천 액션: {analysis['recommended_action']}")
    
    # 신호 확인
    signals_with_action = result["signals"][result["signals"]["signal"] != 0]
    if len(signals_with_action) > 0:
        print("\n생성된 신호 (처음 5개):")
        print(signals_with_action.head())

