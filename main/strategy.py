# main/strategy.py
from typing import Dict, Optional, Union
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
# ---------------------------------------
# 0) Prophet 예측 정렬/준비 헬퍼 함수
# ---------------------------------------
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
    df: 가격 DataFrame (인덱스가 datetime인 상태, 또는 ds 컬럼이 있는 경우)
    prophet_forecast: Prophet 출력 (DataFrame with ds,yhat,yhat_lower,yhat_upper) or pd.Series indexed by ds
    반환: df 인덱스에 맞춘 DataFrame (columns: yhat, yhat_lower, yhat_upper)
    """
    # 가격 쪽에 datetime 열 확보
    price = df.copy()
    if price.index.name is None or not pd.api.types.is_datetime64_any_dtype(price.index):
        # 인덱스가 datetime이 아니면 'ds'컬럼 사용을 기대
        if "ds" in price.columns:
            price = price.set_index("ds")
        else:
            raise ValueError("df의 인덱스가 datetime이 아니고 'ds' 컬럼도 없습니다.")

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

# ---------------------------------------
# 1) generate_signals 개선판 (prophet DataFrame 지원)
# ---------------------------------------
def generate_signals(
        df: pd.DataFrame,
        prophet_forecast: Union[pd.DataFrame, pd.Series],
        params: Optional[Dict] = None,
) -> pd.DataFrame:
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

        # 매수/매도 조건(기존 조건에 불확실성 필터 추가 가능)
        buy_cond = (rel >= pth) and (rsi <= rsi_buy) and (macd_hist > 0)
        sell_cond = (rel <= -pth) and (rsi >= rsi_sell) and (macd_hist < 0)

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
