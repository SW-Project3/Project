"""
main/trading_system.py

indicators.py, risk.py, strategy.py의 모든 기능을 통합한 통합 모듈
- 지표 계산 (RSI, MACD, Bollinger Bands, ATR)
- 리스크 관리 (포지션 크기 계산)
- 전략 신호 생성 (Prophet 예측 기반)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union
from pathlib import Path
import sys

# Prophet 모델 import (선택적)
try:
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from model.prophet_model import ProphetModel
except ImportError:
    ProphetModel = None

# ================================
# 기본 파라미터 설정
# ================================
DEFAULT_PARAMS: Dict = {
    "rsi_period": 14,          # RSI 계산 기간
    "rsi_buy": 30,             # RSI 30 이하 → 매수
    "rsi_sell": 70,            # RSI 70 이상 → 매도

    "macd_short": 12,          # MACD 단기 EMA 기간
    "macd_long": 26,           # MACD 장기 EMA 기간
    "macd_signal": 9,          # MACD 시그널 EMA 기간

    "bb_period": 20,           # 볼린저 밴드 중심선 기간
    "bb_k": 2.0,               # 볼린저 밴드 표준편차 배수

    "atr_period": 14,          # ATR 계산 기간

    "prophet_threshold": 0.005,  # Prophet 예측값이 현재가 대비 ±0.5% 이상일 때만 유효

    "stop_loss_atr_mult": 1.5,  # 손절 기준: ATR × 1.5
    "take_profit_atr_mult": 3.0, # 익절 기준: ATR × 3.0

    "min_hold_bars": 1,         # 같은 포지션 중복 진입 방지용 최소 유지 구간
}


# ============================
# 1. 지표 계산 함수들 (indicators.py)
# ============================

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    상대강도지수 (RSI) 계산
    
    매개변수:
        - series: 종가와 같은 1차원 시계열
        - period: RSI 계산에 쓰이는 기간(기본 14일)
    
    반환값:
        - RSI 값 시리즈 (0~100 범위)
    """
    # 현재 값에서 바로 전값을 뺀 차분을 만듬
    delta = series.diff()
    # delta에서 음수는 0으로 바꾸고, 양수는 그대로 둔다. → 상승폭만 남음
    gain = delta.clip(lower=0)
    # delta에서 양수는 0으로 만들고 음수는 그대로 둔다. → loss는 항상 >= 0
    loss = -delta.clip(upper=0)

    # 평균 상승폭 계산
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    # 평균 하락폭 계산
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))  # 표준 RSI 공식
    return rsi


def macd(series: pd.Series, short: int = 12, long: int = 26, signal: int = 9):
    """
    이동평균 수렴·발산 (MACD) 계산
    
    매개변수:
        - series: 종가 시리즈
        - short/long: 단기/장기 EMA 기간
        - signal: MACD 신호선의 EMA 기간
    
    반환값:
        - (MACD 라인, 신호선, 히스토그램) 튜플
    """
    # adjust=False는 pandas의 EWM에서 지수 가중치의 누적 보정(adjust)를 끄고,
    # 재귀적인 정의(이전 EMA에 현재 값을 반영하는 방식)를 사용하게 해서
    # 일반적인 금융 EMA 정의와 일치시킨다.
    ema_short = series.ewm(span=short, adjust=False).mean()
    ema_long = series.ewm(span=long, adjust=False).mean()
    
    macd_line = ema_short - ema_long  # MACD 라인
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()  # 신호선
    hist = macd_line - signal_line  # 히스토그램 (MACD 라인과 신호선의 차)
    
    return macd_line, signal_line, hist


def bollinger_bands(series: pd.Series, period: int = 20, k: float = 2.0):
    """
    볼린저 밴드 계산
    
    매개변수:
        - series: 종가 시리즈
        - period: 중심선(MA)과 표준편차 계산 기간
        - k: 표준편차 배수
    
    반환값:
        - (중심선, 상단 밴드, 하단 밴드) 튜플
    """
    ma = series.rolling(window=period).mean()  # 단순 이동평균(SMA, 중심선)
    std = series.rolling(window=period).std()  # 동일 기간의 표준편차
    
    upper = ma + k * std  # 상단 밴드 = 중심선 + k×std
    lower = ma - k * std  # 하단 밴드 = 중심선 - k×std
    
    return ma, upper, lower


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    """
    평균진폭범위 (ATR) 계산
    
    매개변수:
        - high: 고가 시리즈
        - low: 저가 시리즈
        - close: 종가 시리즈
        - period: ATR 계산 기간
    
    반환값:
        - ATR 값 시리즈
    """
    prev_close = close.shift(1)  # 이전 종가를 한 칸 아래로 이동
    
    # True Range (TR)를 계산하는 표준 방식
    # TR의 세 후보:
    # 1. high - low (당일 고저 차)
    # 2. |high - prev_close| (당일 고와 전일 종가의 절대차)
    # 3. |low - prev_close| (당일 저와 전일 종가의 절대차)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    
    # TR의 period 기간 단순 이동평균을 계산하여 ATR을 만듬
    atr = tr.rolling(window=period).mean()
    return atr


# ============================
# 2. 리스크 관리 함수들 (risk.py)
# ============================

def position_size(
        balance: float,
        risk_per_trade: float,
        entry_price: float,
        stop_price: float,
        *,
        leverage: float = 1.0,
        min_position_value: float = 1.0,
        max_position_value: Optional[float] = None,
        round_to: Optional[float] = None
) -> float:
    """
    포지션 크기를 **자산 단위(예: BTC)** 기준으로 계산합니다.

    매개변수:
        - balance: 계좌 잔고 (예: USDT)
        - risk_per_trade: 거래당 위험 비율 (예: 0.01 → 잔고의 1%)
        - entry_price: 진입 가격
        - stop_price: 손절가
    옵션:
        - leverage: 레버리지 배율 (기본 1.0)
        - min_position_value: 최소 포지션 가치 (예: 1 USDT)
        - max_position_value: 최대 포지션 가치 (None이면 제한 없음)
        - round_to: 수량 반올림 단위 (예: 0.0001)

    반환값:
        - 매매할 자산의 수량 (예: BTC 0.02)
    """
    if entry_price <= 0:
        raise ValueError("entry_price must be > 0")

    stop_distance = abs(entry_price - stop_price)
    if stop_distance <= 0:
        return 0.0

    if balance <= 0 or risk_per_trade <= 0:
        return 0.0

    # 리스크 금액 = 잔고 × 거래당 위험비율
    risk_amount = balance * risk_per_trade

    # 레버리지 반영
    effective_risk_amount = risk_amount * leverage

    # 포지션 크기 = 리스크 금액 ÷ 스탑 거리
    size = effective_risk_amount / stop_distance

    # 포지션 가치 제한
    pos_value = size * entry_price
    if pos_value < min_position_value:
        return 0.0
    if max_position_value is not None and pos_value > max_position_value:
        size = max_position_value / entry_price

    # 반올림 처리
    if round_to is not None and round_to > 0:
        multiplier = 1.0 / round_to
        size = int(size * multiplier) / multiplier

    return float(max(0.0, size))


def risk_amount_to_size(
        risk_amount: float,
        entry_price: float,
        *,
        round_to: Optional[float] = None
) -> float:
    """
    절대 위험 금액(risk_amount)을 진입가(entry_price) 기준으로
    자산 수량으로 변환합니다.
    """
    if entry_price <= 0:
        raise ValueError("entry_price must be > 0")
    if risk_amount <= 0:
        return 0.0

    size = risk_amount / entry_price

    if round_to is not None and round_to > 0:
        multiplier = 1.0 / round_to
        size = int(size * multiplier) / multiplier

    return float(size)


# ============================
# 3. 전략 함수들 (strategy.py)
# ============================

def compute_indicators(df: pd.DataFrame, params: Optional[Dict] = None) -> pd.DataFrame:
    """
    모든 기술적 지표를 계산하여 DataFrame에 추가
    
    매개변수:
        - df: 'Close', 'High', 'Low' 컬럼이 포함된 DataFrame
        - params: 지표 계산 파라미터 (None이면 DEFAULT_PARAMS 사용)
    
    반환값:
        - RSI, MACD, Bollinger Bands, ATR 컬럼이 추가된 새로운 DataFrame
    """
    if params is None:
        params = DEFAULT_PARAMS

    df = df.copy()

    # 필수 컬럼 확인
    required = {"Close", "High", "Low"}
    if not required.issubset(df.columns):
        raise ValueError(f"데이터프레임에 다음 컬럼이 필요합니다: {required}")

    # RSI 계산
    df["RSI"] = rsi(df["Close"], period=params["rsi_period"])

    # MACD 계산
    macd_line, signal_line, hist = macd(
        df["Close"],
        short=params["macd_short"],
        long=params["macd_long"],
        signal=params["macd_signal"],
    )
    df["MACD"] = macd_line
    df["MACD_Signal"] = signal_line
    df["MACD_Hist"] = hist

    # Bollinger Bands 계산
    ma, upper, lower = bollinger_bands(
        df["Close"], period=params["bb_period"], k=params["bb_k"]
    )
    df["BB_MA"] = ma
    df["BB_Upper"] = upper
    df["BB_Lower"] = lower
    df["BB_Width"] = upper - lower  # 밴드 폭

    # ATR 계산
    df["ATR"] = atr(
        df["High"], df["Low"], df["Close"], period=params["atr_period"]
    )

    return df


def align_prophet_forecast(
        df: pd.DataFrame,
        prophet_forecast: Union[pd.DataFrame, pd.Series],
        ds_col: str = "ds",
        yhat_col: str = "yhat",
        yhat_lower_col: str = "yhat_lower",
        yhat_upper_col: str = "yhat_upper",
        method: str = "forward"  # merge_asof 방향: 'forward' -> 다음 예측을 현재 타임스텝에 매칭
) -> pd.DataFrame:
    """
    Prophet 예측 결과를 가격 데이터의 인덱스에 맞춰 정렬
    
    매개변수:
        - df: 가격 DataFrame (인덱스가 datetime인 상태, 또는 ds 컬럼이 있는 경우)
        - prophet_forecast: Prophet 출력 (DataFrame with ds,yhat,yhat_lower,yhat_upper) or pd.Series indexed by ds
        - method: merge_asof 방향 ('forward', 'backward', 'nearest')
    
    반환값:
        - df 인덱스에 맞춘 DataFrame (columns: yhat, yhat_lower, yhat_upper)
    """
    # 가격 쪽에 datetime 열 확보
    price = df.copy()
    # 인덱스가 datetime인지 확인 (여러 방법으로 체크)
    is_datetime_index = (
        isinstance(price.index, pd.DatetimeIndex) or
        pd.api.types.is_datetime64_any_dtype(price.index) or
        (hasattr(price.index, 'dtype') and pd.api.types.is_datetime64_any_dtype(price.index.dtype))
    )
    
    if not is_datetime_index:
        # 인덱스가 datetime이 아니면 'ds'컬럼 사용을 기대
        if "ds" in price.columns:
            price = price.set_index("ds")
            price.index = pd.to_datetime(price.index)
        else:
            raise ValueError("df의 인덱스가 datetime이 아니고 'ds' 컬럼도 없습니다.")
    else:
        # datetime 인덱스가 확실하도록 변환
        price.index = pd.to_datetime(price.index)

    # prophet_forecast 처리
    if isinstance(prophet_forecast, pd.Series):
        # 시리즈인 경우 인덱스를 datetime으로 만들고 yhat으로 간주
        pf = prophet_forecast.to_frame(name=yhat_col).reset_index().rename(columns={"index": ds_col})
        pf[ds_col] = pd.to_datetime(pf[ds_col])
    else:
        # DataFrame이면 필요한 컬럼만 골라서
        pf = prophet_forecast.copy()
        if ds_col not in pf.columns:
            # maybe index is ds
            if pf.index.name is None:
                raise ValueError("prophet_forecast에 'ds' 컬럼이 없습니다.")
            pf = pf.reset_index().rename(columns={pf.index.name: ds_col})
        pf[ds_col] = pd.to_datetime(pf[ds_col])
        # ensure yhat cols exist (fill with NaN if missing)
        for c in (yhat_col, yhat_lower_col, yhat_upper_col):
            if c not in pf.columns:
                pf[c] = np.nan

    # 정렬
    pf = pf.sort_values(ds_col)
    price_idx = price.index.to_frame(index=False).rename(columns={price.index.name or 0: ds_col})
    price_idx[ds_col] = pd.to_datetime(price_idx[ds_col])

    # merge_asof 를 위해 reset_index
    price_reset = price.reset_index().rename(columns={price.index.name or "index": ds_col})
    price_reset[ds_col] = pd.to_datetime(price_reset[ds_col])

    # merge_asof: price times에 가장 가까운(또는 다음) forecast를 붙임
    merged = pd.merge_asof(
        price_reset.sort_values(ds_col),
        pf.sort_values(ds_col),
        on=ds_col,
        direction=method  # 'forward' 또는 'backward' 또는 'nearest'
    )

    # 결과를 인덱스에 맞춘 DataFrame으로 반환 (인덱스 동일)
    merged = merged.set_index(ds_col)
    res = merged[[yhat_col, yhat_lower_col, yhat_upper_col]]
    res.index = pd.to_datetime(res.index)
    # reindex to exact original index (in case of name issues)
    res = res.reindex(price.index)
    return res


def generate_signals(
        df: pd.DataFrame,
        prophet_forecast: Union[pd.DataFrame, pd.Series],
        params: Optional[Dict] = None,
        track_conditions: bool = True,
) -> pd.DataFrame:
    """
    매매 신호 생성 (Prophet 예측 기반)
    
    매개변수:
        - df: 가격 데이터 (지표 포함 또는 미포함)
        - prophet_forecast: Prophet 예측 결과
        - params: 전략 파라미터
        - track_conditions: 지표 조건 만족 구간 추적 여부 (기본 True)
    
    반환값:
        - 신호 DataFrame (signal, reason, entry_price, stop_loss, take_profit)
          track_conditions=True인 경우 추가 컬럼:
          - condition_prophet_buy, condition_rsi_buy, condition_macd_buy
          - condition_prophet_sell, condition_rsi_sell, condition_macd_sell
          - condition_buy_all, condition_sell_all (모든 조건 만족 여부)
    """
    if params is None:
        params = DEFAULT_PARAMS

    # 지표 계산 보증
    needed = {"RSI", "MACD_Hist", "BB_MA", "ATR"}
    if not needed.issubset(df.columns):
        df = compute_indicators(df, params)

    # Prophet 예측을 가격 인덱스에 맞춰 정렬/병합
    # prophet_forecast는 DataFrame(yhat,yhat_lower,yhat_upper with ds) 또는 Series(index=ds)
    pf_aligned = align_prophet_forecast(df, prophet_forecast, method="forward")

    df = df.copy()
    # join aligned forecast columns
    df = df.join(pf_aligned)

    # 신호 DataFrame 초기화
    signals = pd.DataFrame(index=df.index)
    signals["signal"] = 0
    signals["reason"] = ""
    signals["entry_price"] = np.nan
    signals["stop_loss"] = np.nan
    signals["take_profit"] = np.nan

    # 지표 조건 추적 컬럼 초기화
    if track_conditions:
        signals["condition_prophet_buy"] = False
        signals["condition_rsi_buy"] = False
        signals["condition_macd_buy"] = False
        signals["condition_prophet_sell"] = False
        signals["condition_rsi_sell"] = False
        signals["condition_macd_sell"] = False
        signals["condition_buy_all"] = False
        signals["condition_sell_all"] = False

    # 파라미터
    pth = params["prophet_threshold"]
    rsi_buy = params["rsi_buy"]
    rsi_sell = params["rsi_sell"]
    sl_mult = params["stop_loss_atr_mult"]
    tp_mult = params["take_profit_atr_mult"]

    holding = 0

    for ts in df.index:
        price = df.at[ts, "Close"]
        yhat = df.at[ts, "yhat"]
        yhat_l = df.at[ts, "yhat_lower"]
        yhat_u = df.at[ts, "yhat_upper"]
        rsi = df.at[ts, "RSI"]
        macd_hist = df.at[ts, "MACD_Hist"]
        atr = df.at[ts, "ATR"]

        # 필요한 값 체크
        if pd.isna(price) or pd.isna(yhat) or pd.isna(rsi) or pd.isna(macd_hist) or pd.isna(atr):
            continue

        # 파생값
        rel = (yhat - price) / price
        rel_uncert = np.nan
        if not pd.isna(yhat) and not pd.isna(yhat_l) and not pd.isna(yhat_u) and yhat != 0:
            rel_uncert = (yhat_u - yhat_l) / abs(yhat)

        # 각 지표 조건 개별 체크
        prophet_buy_cond = rel >= pth
        rsi_buy_cond = rsi <= rsi_buy
        macd_buy_cond = macd_hist > 0
        
        prophet_sell_cond = rel <= -pth
        rsi_sell_cond = rsi >= rsi_sell
        macd_sell_cond = macd_hist < 0

        # 전체 조건
        buy_cond = prophet_buy_cond and rsi_buy_cond and macd_buy_cond
        sell_cond = prophet_sell_cond and rsi_sell_cond and macd_sell_cond

        # 지표 조건 추적
        if track_conditions:
            signals.at[ts, "condition_prophet_buy"] = prophet_buy_cond
            signals.at[ts, "condition_rsi_buy"] = rsi_buy_cond
            signals.at[ts, "condition_macd_buy"] = macd_buy_cond
            signals.at[ts, "condition_prophet_sell"] = prophet_sell_cond
            signals.at[ts, "condition_rsi_sell"] = rsi_sell_cond
            signals.at[ts, "condition_macd_sell"] = macd_sell_cond
            signals.at[ts, "condition_buy_all"] = buy_cond
            signals.at[ts, "condition_sell_all"] = sell_cond

        # 예: 불확실성이 크면 신호 억제 (임계값은 필요시 params로 노출)
        unc_thresh = params.get("prophet_uncertainty_threshold", 0.3)
        if rel_uncert is not np.nan and rel_uncert > unc_thresh:
            # 너무 불확실하면 스킵
            continue

        if buy_cond and holding <= 0:
            signals.at[ts, "signal"] = 1
            signals.at[ts, "reason"] = f"Prophet 상승({rel:.3f}) unc={rel_uncert:.3f} RSI={rsi:.1f} MACD={macd_hist:.6f}"
            signals.at[ts, "entry_price"] = price
            signals.at[ts, "stop_loss"] = price - sl_mult * atr
            signals.at[ts, "take_profit"] = price + tp_mult * atr
            holding = 1

        elif sell_cond and holding >= 0:
            signals.at[ts, "signal"] = -1
            signals.at[ts, "reason"] = f"Prophet 하락({rel:.3f}) unc={rel_uncert:.3f} RSI={rsi:.1f} MACD={macd_hist:.6f}"
            signals.at[ts, "entry_price"] = price
            signals.at[ts, "stop_loss"] = price + sl_mult * atr
            signals.at[ts, "take_profit"] = price - tp_mult * atr
            holding = -1

    return signals


def analyze_signal_conditions(
        signals: pd.DataFrame,
        df: Optional[pd.DataFrame] = None,
        lookback_periods: int = 10
) -> pd.DataFrame:
    """
    신호 생성 시점에서 각 지표 조건이 만족된 구간을 분석
    
    매개변수:
        - signals: generate_signals()로 생성된 신호 DataFrame
        - df: 원본 가격 데이터 (선택적, 지표 값 확인용)
        - lookback_periods: 신호 생성 전 몇 개 기간을 분석할지 (기본 10)
    
    반환값:
        - 각 신호에 대한 조건 만족 구간 분석 DataFrame
    """
    if "condition_buy_all" not in signals.columns:
        raise ValueError("signals에 지표 조건 추적 정보가 없습니다. generate_signals(..., track_conditions=True)로 생성하세요.")
    
    analysis_list = []
    
    # 매수 신호 분석
    buy_signals = signals[signals["signal"] == 1]
    for signal_ts in buy_signals.index:
        # 신호 생성 시점 이전 구간 분석
        signal_idx = signals.index.get_loc(signal_ts)
        start_idx = max(0, signal_idx - lookback_periods)
        analysis_window = signals.iloc[start_idx:signal_idx + 1]
        
        # 각 조건이 만족된 기간 수 계산
        prophet_satisfied = analysis_window["condition_prophet_buy"].sum()
        rsi_satisfied = analysis_window["condition_rsi_buy"].sum()
        macd_satisfied = analysis_window["condition_macd_buy"].sum()
        all_satisfied = analysis_window["condition_buy_all"].sum()
        
        analysis_list.append({
            "timestamp": signal_ts,
            "signal_type": "buy",
            "prophet_condition_periods": prophet_satisfied,
            "rsi_condition_periods": rsi_satisfied,
            "macd_condition_periods": macd_satisfied,
            "all_conditions_periods": all_satisfied,
            "total_periods": len(analysis_window),
            "entry_price": signals.at[signal_ts, "entry_price"],
            "reason": signals.at[signal_ts, "reason"]
        })
    
    # 매도 신호 분석
    sell_signals = signals[signals["signal"] == -1]
    for signal_ts in sell_signals.index:
        # 신호 생성 시점 이전 구간 분석
        signal_idx = signals.index.get_loc(signal_ts)
        start_idx = max(0, signal_idx - lookback_periods)
        analysis_window = signals.iloc[start_idx:signal_idx + 1]
        
        # 각 조건이 만족된 기간 수 계산
        prophet_satisfied = analysis_window["condition_prophet_sell"].sum()
        rsi_satisfied = analysis_window["condition_rsi_sell"].sum()
        macd_satisfied = analysis_window["condition_macd_sell"].sum()
        all_satisfied = analysis_window["condition_sell_all"].sum()
        
        analysis_list.append({
            "timestamp": signal_ts,
            "signal_type": "sell",
            "prophet_condition_periods": prophet_satisfied,
            "rsi_condition_periods": rsi_satisfied,
            "macd_condition_periods": macd_satisfied,
            "all_conditions_periods": all_satisfied,
            "total_periods": len(analysis_window),
            "entry_price": signals.at[signal_ts, "entry_price"],
            "reason": signals.at[signal_ts, "reason"]
        })
    
    if analysis_list:
        analysis_df = pd.DataFrame(analysis_list)
        analysis_df = analysis_df.set_index("timestamp")
        return analysis_df
    else:
        return pd.DataFrame()


def get_condition_satisfaction_periods(
        signals: pd.DataFrame,
        signal_timestamp: pd.Timestamp,
        signal_type: str = "buy",
        max_periods: int = 100
) -> Dict:
    """
    특정 신호 시점에서 각 지표 조건이 연속으로 만족된 기간을 계산
    
    매개변수:
        - signals: generate_signals()로 생성된 신호 DataFrame
        - signal_timestamp: 분석할 신호의 타임스탬프
        - signal_type: "buy" 또는 "sell"
        - max_periods: 최대 역추적 기간 수
    
    반환값:
        - 각 조건의 연속 만족 기간 정보를 담은 딕셔너리
    """
    if signal_timestamp not in signals.index:
        raise ValueError(f"신호 타임스탬프 {signal_timestamp}가 signals에 없습니다.")
    
    signal_idx = signals.index.get_loc(signal_timestamp)
    start_idx = max(0, signal_idx - max_periods)
    window = signals.iloc[start_idx:signal_idx + 1]
    
    if signal_type == "buy":
        prophet_col = "condition_prophet_buy"
        rsi_col = "condition_rsi_buy"
        macd_col = "condition_macd_buy"
    else:
        prophet_col = "condition_prophet_sell"
        rsi_col = "condition_rsi_sell"
        macd_col = "condition_macd_sell"
    
    # 역순으로 순회하여 연속 만족 기간 계산
    prophet_streak = 0
    rsi_streak = 0
    macd_streak = 0
    
    for i in range(len(window) - 1, -1, -1):
        if window.iloc[i][prophet_col]:
            prophet_streak += 1
        else:
            break
    
    for i in range(len(window) - 1, -1, -1):
        if window.iloc[i][rsi_col]:
            rsi_streak += 1
        else:
            break
    
    for i in range(len(window) - 1, -1, -1):
        if window.iloc[i][macd_col]:
            macd_streak += 1
        else:
            break
    
    return {
        "signal_timestamp": signal_timestamp,
        "signal_type": signal_type,
        "prophet_condition_streak": prophet_streak,
        "rsi_condition_streak": rsi_streak,
        "macd_condition_streak": macd_streak,
        "min_streak": min(prophet_streak, rsi_streak, macd_streak),
        "max_streak": max(prophet_streak, rsi_streak, macd_streak)
    }


def apply_signals_simple_backtest(
        df: pd.DataFrame,
        signals: pd.DataFrame,
        initial_cash: float = 10000.0,
        risk_per_trade: float = 0.01,
) -> pd.DataFrame:
    """
    아주 단순한 백테스트 시뮬레이터

    - 신호 발생 시 종가 기준으로 매매 (마켓온클로즈 가정)
    - 포지션은 한 번에 1개만 (중복 진입 없음)
    - 손절/익절가는 신호에 기록된 값 사용
    - 수수료, 슬리피지 등은 고려하지 않음
    """
    df = df.copy()
    signals = signals.reindex(df.index).fillna(0)

    cash = initial_cash
    position = 0.0
    entry_price = None
    stop_loss = None
    take_profit = None

    nav = pd.Series(index=df.index, dtype=float)  # 순자산 그래프용

    for ts in df.index:
        price = df.at[ts, "Close"]
        sig = int(signals.at[ts, "signal"])

        # 포지션이 있을 때 손절/익절 조건 체크
        if position != 0.0:
            if position > 0:  # 롱 포지션
                if price <= stop_loss or price >= take_profit:
                    pnl = (price - entry_price) * position
                    cash += pnl
                    position = 0.0
            else:  # 숏 포지션
                if price >= stop_loss or price <= take_profit:
                    pnl = (entry_price - price) * abs(position)
                    cash += pnl
                    position = 0.0

        # 새 신호 발생 시 진입
        if sig != 0 and position == 0.0:
            entry_price = price
            stop_loss = signals.at[ts, "stop_loss"]
            take_profit = signals.at[ts, "take_profit"]

            # 리스크 기반 포지션 사이즈 계산
            risk_amount = cash * risk_per_trade
            distance = abs(entry_price - stop_loss)
            size = risk_amount / distance if distance > 0 else 0
            position = size * sig

        # 현재 순자산 계산
        nav.at[ts] = cash + position * price

    df["NAV"] = nav
    return df


# ============================
# 4. Prophet 연결 함수들 (prophet_strategy_connector.py)
# ============================

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
    prophet_model,
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
    prophet_model: Optional = None,
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
):
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
    prophet_model: Optional = None,
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
    # 통합 테스트
    print("=== Trading System 통합 테스트 ===\n")
    
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
    
    print("1. 지표 계산 테스트...")
    df_with_indicators = compute_indicators(demo_df)
    print(f"   ✓ 지표 계산 완료: {len(df_with_indicators.columns)} 컬럼")
    
    print("\n2. 리스크 관리 테스트...")
    pos_size = position_size(10000, 0.01, entry_price=50000, stop_price=49000)
    print(f"   ✓ 포지션 크기 계산: {pos_size:.6f}")
    
    print("\n3. 전략 신호 생성 테스트...")
    # Prophet 예측 더미 데이터
    prophet_forecast = pd.DataFrame({
        "ds": dates,
        "yhat": prices * 1.01,
        "yhat_lower": prices * 0.99,
        "yhat_upper": prices * 1.03
    })
    
    signals = generate_signals(df_with_indicators, prophet_forecast)
    signal_count = len(signals[signals["signal"] != 0])
    print(f"   ✓ 신호 생성 완료: {signal_count} 개의 신호")
    
    print("\n4. 백테스트 테스트...")
    result_df = apply_signals_simple_backtest(df_with_indicators, signals)
    final_nav = result_df["NAV"].iloc[-1]
    print(f"   ✓ 백테스트 완료: 최종 NAV {final_nav:.2f}")
    
    print("\n5. Prophet 연결 테스트...")
    if ProphetModel is not None:
        try:
            prophet_model = create_prophet_model_from_config(
                freq="1H",
                horizon=3,
                start_date="2025-01-01"
            )
            if prophet_model:
                # 전체 파이프라인 실행
                pipeline_result = run_full_pipeline(
                    demo_df,
                    prophet_model=prophet_model,
                    use_rolling_forecast=True,
                    rolling_window=150
                )
                print(f"   ✓ Prophet 파이프라인 완료: {len(pipeline_result['prophet_forecast'])} 예측")
            else:
                print("   ⚠ ProphetModel 생성 실패")
        except Exception as e:
            print(f"   ⚠ Prophet 테스트 스킵: {e}")
    else:
        print("   ⚠ ProphetModel을 사용할 수 없습니다 (선택적 기능)")
    
    print("\n=== 모든 기능 통합 완료! ===")

