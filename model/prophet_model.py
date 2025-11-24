# prophet_model.py
"""
Prophet 예측 엔진

[세팅]
- 봉 단위: 1일("1d")
- 학습 범위: 조정가능 (최소 2023-01-01 부터)
- 예측 스텝: horizon=3 (다음 캔들 3개)
- 회귀자:
    - volume: 거래량(추세의 신뢰도 반영)
    - crash_dummy: 급등락/볼륨 스파이크 이벤트 강도(0~1)
    - rsi14(옵션): 14일 RSI, 과매수·과매도 구간을 보조적으로 반영

[입력]
- pandas.DataFrame
- 인덱스: DatetimeIndex (UTC, 정렬, 중복 없음)
- 컬럼: "close"(float), "volume"(float)

[출력]
- dict:
    {
      "yhat": float,          # horizon 구간(yhat)의 '평균' (신호용 대표값)
      "yhat_lower": float,    # horizon 구간 yhat_lower의 '최소값' (보수적 하단)
      "yhat_upper": float,    # horizon 구간 yhat_upper의 '최대값' (보수적 상단)
    }
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Literal, List
import numpy as np
import pandas as pd
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)


# Prophet import 체크
try:
    from prophet import Prophet
except Exception as e:
    Prophet = None
    _IMPORT_ERROR = e

# 전략 쪽 보조지표 모듈 (RSI 재사용)
try:
    from main import indicators
except Exception as e:
    indicators = None
    _INDICATOR_IMPORT_ERROR = e


@dataclass
class ProphetModel:
    # ====== 핵심 하이퍼파라미터 ======
    freq: str = "1d"                      # 예측할 주기
    horizon: int = 3                      # 예측 스텝 수
    start_date: str = "2025-04-01"        # 학습 시작일(UTC)
    end_date: Optional[str] = "2025-10-30T00:09:00+00:00"  # 학습 종료일(UTC)
    interval_width: float = 0.80          # 신뢰구간 폭
    changepoint_prior_scale: float = 1.0  # 추세 변곡점 민감도
    seasonality_mode: Literal["additive", "multiplicative"] = "additive"

    # 급변 감지 기준
    ret_spike_thresh: float = 0.08
    vol_z_window: int = 60
    vol_z_thresh: float = 2.0
    crash_decay_bars: int = 1

    # 휴일 이벤트 관련
    use_fomc_holidays: bool = True
    holidays_prior_scale: float = 3.0
    pre_event_days: int = 1

    # 로그-가격, 매물대, 튜닝
    use_log_price: bool = True
    n_changepoints: int = 35
    changepoint_range: float = 0.9

    # RSI (옵션)
    use_rsi: bool = True
    rsi_period: int = 7

    # 매물대(VAP) 관련
    use_vap: bool = False
    vap_window: int = 30
    vap_bins: int = 60
    vap_smooth_sigma: float = 0.15

    # ====== 내부 상태(로그/디버깅용) ======
    _last_fit_samples: int = field(init=False, default=0)

    # ---------------------------
    # 0) 환경 점검
    # ---------------------------
    def _check_env(self) -> bool:
        if Prophet is None:
            print(f"[ProphetModel] ImportError: {getattr(_IMPORT_ERROR, 'msg', _IMPORT_ERROR)}")
            return False
        if self.use_rsi and indicators is None:
            print(f"[ProphetModel] indicators import failed, cannot use RSI: {_INDICATOR_IMPORT_ERROR}")
        return True

    # ---------------------------
    # 1) 입력 검증 + 최소 정리
    # ---------------------------
    def _validate_input(self, df_in: pd.DataFrame) -> Optional[pd.DataFrame]:
        if not isinstance(df_in, pd.DataFrame):
            print("[ProphetModel][validate] 입력은 DataFrame이어야 합니다.")
            return None

        required = {"close", "volume"}
        missing = [c for c in required if c not in df_in.columns]
        if missing:
            print(f"[ProphetModel][validate] 필수 컬럼 누락: {missing}")
            return None

        df = df_in.copy()

        if not isinstance(df.index, pd.DatetimeIndex):
            print("[ProphetModel][validate] DatetimeIndex가 필요합니다.")
            return None

        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        df = df[~df.index.duplicated(keep="last")].sort_index()

        for c in required:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=list(required), inplace=True)

        df = df[(df["close"] > 0) & (df["volume"] >= 0)]

        try:
            start_ts = pd.Timestamp(self.start_date)
        except Exception:
            start_ts = pd.to_datetime(str(self.start_date))
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        else:
            start_ts = start_ts.tz_convert("UTC")
        df = df[df.index >= start_ts]

        if self.end_date is not None:
            try:
                end_ts = pd.Timestamp(self.end_date)
            except Exception:
                end_ts = pd.to_datetime(str(self.end_date))
            if end_ts.tzinfo is None:
                end_ts = end_ts.tz_localize("UTC")
            else:
                end_ts = end_ts.tz_convert("UTC")
            df = df[df.index <= end_ts]

        if len(df) < 200:
            print(f"[ProphetModel][validate] 샘플이 적을 수 있음: len={len(df)}")
        return df

    # ---------------------------
    # 2) 급변/볼륨스파이크 감지
    # ---------------------------
    def _build_event_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        ret = out["close"].pct_change().fillna(0.0)
        ret_flag = (ret.abs() > self.ret_spike_thresh).astype(int)

        w = max(self.vol_z_window, 20)
        vol = out["volume"].astype(float)
        vol_mean = vol.rolling(w, min_periods=20).mean()
        vol_std = vol.rolling(w, min_periods=20).std()
        den = vol_std.replace(0, np.nan)
        vol_z = (vol - vol_mean) / den
        vol_z = vol_z.fillna(0.0)
        vol_flag = (vol_z > self.vol_z_thresh).astype(int)

        base = ((ret_flag == 1) | (vol_flag == 1)).astype(int).values

        decay = np.zeros_like(base, dtype=float)
        for i in range(len(base)):
            if base[i] == 1:
                decay[i] += 1.0
                for k in range(1, self.crash_decay_bars + 1):
                    j = i + k
                    if j < len(decay):
                        decay[j] = max(decay[j], 1.0 / (2 ** k))

        out["crash_dummy"] = decay
        return out

    # ---------------------------
    # 3) Prophet 입력 포맷 변환
    # ---------------------------
    def _to_prophet_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        flg = self._build_event_flags(df)

        # RSI 14 계산 (옵션)
        if self.use_rsi and indicators is not None:
            try:
                flg["rsi14"] = indicators.rsi(flg["close"], period=self.rsi_period)
                flg["rsi14"] = flg["rsi14"].ffill().bfill()
            except Exception as e:
                print(f"[ProphetModel][RSI] failed to compute rsi14: {e}")

        if self.use_vap:
            flg = self._build_vap_features(flg)

        ds = flg.index.tz_convert(None)

        pdf_dict: Dict[str, np.ndarray] = {
            "ds": ds,
            "y": np.log(flg["close"].values) if self.use_log_price else flg["close"].values,
            "volume": flg["volume"].values,
            "crash_dummy": flg["crash_dummy"].values,
        }

        if self.use_rsi and "rsi14" in flg.columns:
            pdf_dict["rsi14"] = flg["rsi14"].values

        pdf = pd.DataFrame(pdf_dict)

        if self.use_vap:
            pdf["vap_poc_diff"] = flg["vap_poc_diff"].values
            pdf["vap_val_dist"] = flg["vap_val_dist"].values
            pdf["vap_vah_dist"] = flg["vap_vah_dist"].values

        check_cols = ["ds", "y", "volume", "crash_dummy"]
        if self.use_rsi and "rsi14" in pdf.columns:
            check_cols.append("rsi14")

        if pdf[check_cols].isna().any().any():
            print("[ProphetModel][transform] pdf에 NaN 존재 → 중단")
            return pd.DataFrame()

        return pdf

    # ---------------------------
    # 4) 휴일 효과(FOMC)
    # ---------------------------
    def _build_holidays_df(self) -> pd.DataFrame:
        dates_2025 = [
            "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
            "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
        ]
        ds = pd.to_datetime(dates_2025, utc=True)
        holidays_df = pd.DataFrame(
            {
                "holiday": "FOMC",
                "ds": ds.tz_convert(None),
                "lower_window": -int(self.pre_event_days),
                "upper_window": 0,
            }
        )
        start_ts = pd.Timestamp(self.start_date, tz="UTC")
        end_ts = pd.Timestamp(self.end_date, tz="UTC") if self.end_date else None
        mask = (ds >= start_ts) & ((end_ts is None) | (ds <= end_ts))
        holidays_df = holidays_df.loc[mask]
        return holidays_df.reset_index(drop=True)

    # ---------------------------
    # VAP features
    # ---------------------------
    def _build_vap_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.use_vap or len(df) < max(self.vap_window, 30):
            out = df.copy()
            out["vap_poc_diff"] = 0.0
            out["vap_val_dist"] = 0.0
            out["vap_vah_dist"] = 0.0
            return out

        try:
            from scipy.ndimage import gaussian_filter1d
        except Exception:
            gaussian_filter1d = None

        out = df.copy()
        tail = out.iloc[-self.vap_window:]
        prices = tail["close"].astype(float).values
        volumes = tail["volume"].astype(float).values

        p_min, p_max = prices.min(), prices.max()
        if p_min == p_max:
            out["vap_poc_diff"] = 0.0
            out["vap_val_dist"] = 0.0
            out["vap_vah_dist"] = 0.0
            return out

        bin_edges = np.linspace(p_min, p_max, num=self.vap_bins + 1)
        hist, edges = np.histogram(prices, bins=bin_edges, weights=volumes)

        if self.vap_smooth_sigma and self.vap_smooth_sigma > 0 and gaussian_filter1d is not None:
            hist = gaussian_filter1d(hist.astype(float), sigma=self.vap_smooth_sigma)

        bin_centers = (edges[:-1] + edges[1:]) / 2.0
        poc_price = float(bin_centers[np.argmax(hist)])

        cum = hist.cumsum()
        total = cum[-1] if cum[-1] > 0 else 1.0
        lower_cut = total * 0.15
        upper_cut = total * 0.85
        val_price = float(bin_centers[np.searchsorted(cum, lower_cut, side="left")])
        vah_price = float(bin_centers[np.searchsorted(cum, upper_cut, side="left")])

        last_close = float(out["close"].iloc[-1])
        out["vap_poc_diff"] = (last_close - poc_price) / last_close
        out["vap_val_dist"] = (last_close - val_price) / last_close
        out["vap_vah_dist"] = (last_close - vah_price) / last_close

        out[["vap_poc_diff", "vap_val_dist", "vap_vah_dist"]] = (
            out[["vap_poc_diff", "vap_val_dist", "vap_vah_dist"]].ffill().bfill()
        )

        return out

    # ---------------------------
    # 5) 학습 + 멀티스텝 예측
    # ---------------------------
    def fit_predict(self, df: pd.DataFrame) -> Optional[Dict]:
        if not self._check_env():
            return None

        dfv = self._validate_input(df)
        if dfv is None or len(dfv) == 0:
            return None

        self._last_fit_samples = len(dfv)

        pdf = self._to_prophet_frame(dfv)
        if pdf.empty:
            return None

        holidays_df = None
        if self.use_fomc_holidays:
            holidays_df = self._build_holidays_df()

        m = Prophet(
            interval_width=self.interval_width,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_mode=self.seasonality_mode,
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            n_changepoints=self.n_changepoints,
            changepoint_range=self.changepoint_range,
            seasonality_prior_scale=1.0,
            holidays=holidays_df,
            holidays_prior_scale=self.holidays_prior_scale,
        )

        m.add_regressor("volume", standardize=True, prior_scale=0.05)
        m.add_regressor("crash_dummy", standardize=True, prior_scale=0.3)

        if self.use_rsi and "rsi14" in pdf.columns:
            m.add_regressor("rsi14", standardize=True, prior_scale=0.1)

        if self.use_vap:
            m.add_regressor("vap_poc_diff", standardize=True, prior_scale=0.2)
            m.add_regressor("vap_val_dist", standardize=True, prior_scale=0.1)
            m.add_regressor("vap_vah_dist", standardize=True, prior_scale=0.1)

        try:
            m.fit(pdf)
        except Exception as e:
            print(f"[ProphetModel][fit] 실패: {e}")
            return None

        try:
            future = m.make_future_dataframe(
                periods=self.horizon,
                freq=self.freq,
                include_history=False,
            )

            k = min(14, len(dfv))
            hist_vol = dfv["volume"].iloc[-k:]
            p10, p90 = hist_vol.quantile([0.10, 0.90])
            vol_med = float(hist_vol.median())
            future["volume"] = float(np.clip(vol_med, p10, p90))

            future["crash_dummy"] = 0.0

            # 미래 rsi14: 마지막 값을 그대로 사용(현재 모멘텀 유지 가정)
            if self.use_rsi and "rsi14" in pdf.columns:
                last_rsi = float(pdf["rsi14"].iloc[-1])
                future["rsi14"] = last_rsi

            if self.use_vap:
                future["vap_poc_diff"] = float(pdf["vap_poc_diff"].iloc[-1])
                future["vap_val_dist"] = float(pdf["vap_val_dist"].iloc[-1])
                future["vap_vah_dist"] = float(pdf["vap_vah_dist"].iloc[-1])

            fcst = m.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]

            if self.use_log_price:
                fcst["yhat"] = np.exp(fcst["yhat"])
                fcst["yhat_lower"] = np.exp(fcst["yhat_lower"])
                fcst["yhat_upper"] = np.exp(fcst["yhat_upper"])

            last_close = float(dfv["close"].iloc[-1])

            def _soft_attenuate(
                    fcst_: pd.DataFrame, last_close_: float, k_: float = 4.0
            ) -> pd.DataFrame:
                out = fcst_.copy()
                delta = out["yhat"] - last_close_
                delta_pct = delta / last_close_

                weight = 1.0 / (1.0 + k_ * np.abs(delta_pct))

                out["yhat"] = last_close_ + delta * weight

                low_delta = out["yhat_lower"] - last_close_
                up_delta = out["yhat_upper"] - last_close_
                out["yhat_lower"] = last_close_ + low_delta * weight
                out["yhat_upper"] = last_close_ + up_delta * weight

                lower = out[["yhat_lower", "yhat_upper"]].min(axis=1)
                upper = out[["yhat_lower", "yhat_upper"]].max(axis=1)
                out["yhat_lower"] = lower
                out["yhat_upper"] = upper

                return out

            fcst = _soft_attenuate(fcst, last_close, k_=4.0)

            yhat_avg = float(fcst["yhat"].mean())
            yhat_low = float(fcst["yhat_lower"].min())
            yhat_up = float(fcst["yhat_upper"].max())

            result = {
                "yhat": yhat_avg,
                "yhat_lower": yhat_low,
                "yhat_upper": yhat_up,
            }
            return result

        except Exception as e:
            print(f"[ProphetModel][predict] 실패: {e}")
            return None