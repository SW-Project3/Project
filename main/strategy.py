# main/strategy.py
from typing import Dict, Optional
import pandas as pd
import numpy as np

from main import indicators

# ===============================
# 기본 파라미터 설정 (필요시 configs/params.yaml로 옮겨서 관리 가능)
# ===============================
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


# ===================================
# 1) 보조지표 계산 함수
# ===================================
def compute_indicators(df: pd.DataFrame, params: Optional[Dict] = None) -> pd.DataFrame:
    """
    df: 'Close', 'High', 'Low' 컬럼이 포함된 DataFrame
    반환: RSI, MACD, Bollinger Bands, ATR 컬럼이 추가된 새로운 DataFrame
    """
    if params is None:
        params = DEFAULT_PARAMS

    df = df.copy()

    # 필수 컬럼 확인
    required = {"Close", "High", "Low"}
    if not required.issubset(df.columns):
        raise ValueError(f"데이터프레임에 다음 컬럼이 필요합니다: {required}")

    # RSI 계산
    df["RSI"] = indicators.rsi(df["Close"], period=params["rsi_period"])

    # MACD 계산
    macd_line, signal_line, hist = indicators.macd(
        df["Close"],
        short=params["macd_short"],
        long=params["macd_long"],
        signal=params["macd_signal"],
    )
    df["MACD"] = macd_line
    df["MACD_Signal"] = signal_line
    df["MACD_Hist"] = hist

    # Bollinger Bands 계산
    ma, upper, lower = indicators.bollinger_bands(
        df["Close"], period=params["bb_period"], k=params["bb_k"]
    )
    df["BB_MA"] = ma
    df["BB_Upper"] = upper
    df["BB_Lower"] = lower
    df["BB_Width"] = upper - lower  # 밴드 폭

    # ATR 계산
    df["ATR"] = indicators.atr(
        df["High"], df["Low"], df["Close"], period=params["atr_period"]
    )

    return df


# ===================================
# 2) 매매 신호 생성 함수
# ===================================
def generate_signals(
        df: pd.DataFrame,
        prophet_forecast: pd.Series,
        params: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Prophet 예측값과 보조지표를 결합해서 매매 신호 생성

    매수 조건 (예시):
        - Prophet 예측값이 현재가보다 0.5% 이상 높을 때
        - RSI가 30 이하 (과매도)
        - MACD 히스토그램이 0 이상 (상승 모멘텀)

    매도 조건 (예시):
        - Prophet 예측값이 현재가보다 0.5% 이상 낮을 때
        - RSI가 70 이상 (과매수)
        - MACD 히스토그램이 0 이하 (하락 모멘텀)
    """
    if params is None:
        params = DEFAULT_PARAMS

    # Prophet 예측값과 인덱스 맞추기
    forecast = prophet_forecast.reindex(df.index)
    df = df.copy()

    # 지표가 없으면 계산
    needed = {"RSI", "MACD_Hist", "BB_MA", "ATR"}
    if not needed.issubset(df.columns):
        df = compute_indicators(df, params)

    # 신호 저장용 DataFrame 생성
    signals = pd.DataFrame(index=df.index)
    signals["signal"] = 0           # 1 = 매수, -1 = 매도, 0 = 없음
    signals["reason"] = ""          # 신호 발생 이유
    signals["entry_price"] = np.nan
    signals["stop_loss"] = np.nan
    signals["take_profit"] = np.nan

    # 파라미터 불러오기
    pth = params["prophet_threshold"]
    rsi_buy = params["rsi_buy"]
    rsi_sell = params["rsi_sell"]
    sl_mult = params["stop_loss_atr_mult"]
    tp_mult = params["take_profit_atr_mult"]

    holding = 0  # 현재 포지션 상태 (0: 없음, 1: 롱, -1: 숏)

    for ts in df.index:
        price = df.at[ts, "Close"]
        f = forecast.at[ts]
        rsi = df.at[ts, "RSI"]
        macd_hist = df.at[ts, "MACD_Hist"]
        atr = df.at[ts, "ATR"]

        # 데이터가 비어있으면 건너뛰기
        if pd.isna(price) or pd.isna(f) or pd.isna(rsi) or pd.isna(macd_hist) or pd.isna(atr):
            continue

        # Prophet 예측 대비 상승/하락 비율
        rel = (f - price) / price

        # 매수 조건
        buy_cond = (rel >= pth) and (rsi <= rsi_buy) and (macd_hist > 0)
        # 매도 조건
        sell_cond = (rel <= -pth) and (rsi >= rsi_sell) and (macd_hist < 0)

        # 매수 신호
        if buy_cond and holding <= 0:
            signals.at[ts, "signal"] = 1
            signals.at[ts, "reason"] = f"Prophet 상승({rel:.3f}) + RSI({rsi:.1f}) + MACD({macd_hist:.6f})"
            signals.at[ts, "entry_price"] = price
            signals.at[ts, "stop_loss"] = price - sl_mult * atr
            signals.at[ts, "take_profit"] = price + tp_mult * atr
            holding = 1

        # 매도 신호
        elif sell_cond and holding >= 0:
            signals.at[ts, "signal"] = -1
            signals.at[ts, "reason"] = f"Prophet 하락({rel:.3f}) + RSI({rsi:.1f}) + MACD({macd_hist:.6f})"
            signals.at[ts, "entry_price"] = price
            signals.at[ts, "stop_loss"] = price + sl_mult * atr
            signals.at[ts, "take_profit"] = price - tp_mult * atr
            holding = -1

    return signals


# ===================================
# 3) 간단한 백테스트 함수
# ===================================
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
