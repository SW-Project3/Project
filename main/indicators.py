import pandas as pd
"""pd.Series, pd.DataFrame 연산(rolling, shift 등)을 사용하므로 pandas가 필수"""
import numpy as n
"""numpy는 수치 연산(예: np.nan)이나 배열 연산이 필요할 때 사용"""

# ============================
# 보조지표 모음
# ============================

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """series-->종가와 같은 1차원 시계열을 받음"""
    """period-->RSI 계산에 쓰이는 기간(기본 14일)"""
    """상대강도지수 (RSI)"""
    """현재 값에서 바로 전값을 뺀 차분을 만듬"""
    delta = series.diff()
    gain = delta.clip(lower=0)
    """delta에서 음수는 0으로 바꾸고, 양수는 그대로 둔다.-->상승폭만 남음"""
    loss = -delta.clip(upper=0)
    """delta에서 양수는 0으로 만들고 음수는 그대로 둔다.-->loss는 항상 >=이다."""

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    """평균 상승폭 계산"""
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    """평균 하락폭 계산"""

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    """표준 RSI공식"""
    return rsi


def macd(series: pd.Series, short: int = 12, long: int = 26, signal: int = 9):
    """short/long = 단기/장기 EMA기간, signal = MACD 신호선의 EMA기간"""
    """EMA = 지수이동평균"""
    """이동평균 수렴·발산 (MACD)"""
    ema_short = series.ewm(span=short, adjust=False).mean()
    ema_long = series.ewm(span=long, adjust=False).mean()
    """adjust=false는 pandas의 EWM에서 지수 가중치의 누적 보정(adjust)를 끄고, 재귀적인
    정의(이전 EMA에 현재 값을 반영하는 방식)를 사용하게 해서 일반적인 금융 EMA 정의와 
    일치시킨다."""
    macd_line = ema_short - ema_long
    """시계열"""
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    """MACD 라인과 신호선의 차. 히스토그램은 추세의 강도를 시각적으로 확인하는데 쓰인다."""
    return macd_line, signal_line, hist
"""3개의 pd.Series를 튜플로 반환, index는 원래 입력 시계열과 동일하다."""


def bollinger_bands(series: pd.Series, period: int = 20, k: float = 2.0):
    """period = 중심선(MA)과 표준편차 계산 기간"""
    """K = 표준변차 배수"""
    """볼린저 밴드"""
    ma = series.rolling(window=period).mean()
    """단순 이동평균(SMA, 중심선)을 계산"""
    std = series.rolling(window=period).std()
    """동일 기간의 표준편차를 계산"""
    upper = ma + k * std
    lower = ma - k * std
    """상/하 밴드를 계산, 상단 = 중심선 + kstd, 하단 = 중심선 - kstd"""
    return ma, upper, lower


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    """평균진폭범위 (ATR)"""
    prev_close = close.shift(1)
    """이전 종가를 한 칸 아래로 이동시켜서(shift) close[i-1] 값을 현재 인덱스에 맞춤"""
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    """True Range (TR)를 계산하는 표준 방식 
    TR의 세 후보:
    1. high - low (당일 고저 차)
    2. high - prev_close (당일 고와 전일 종가의 절대차)
    3. low - prev_close (당일 저와 전일 종가의 절대차)
    
    pd.concat([...], axis=1)로 세 시리즈를 열로 붙이고 max(axis=1)로
    각 행별 최대값을 취하면 TR이 된다."""

    atr = tr.rolling(window=period).mean()
    """TR의 period 기간 단순 이동평균을 계산하여 ATR을 만듬"""
    return atr
