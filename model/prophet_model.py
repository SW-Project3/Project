# prophet_model.py
"""
Prophet 예측 엔진

[세팅]
- 봉 단위: 1일("1d")
- 학습 범위: 조정가능 (최소 2023-01-01 부터)
- 예측 스텝: horizon=3 (다음 캔들 3개)
- 회귀자: volume(거래량), crash_dummy(급변·볼륨스파이크 이벤트)

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

[동작 개요]
- 1) _validate_input: 타입/인덱스/컬럼/음수/NaN/기간 필터/정렬/중복 제거
- 2) _build_event_flags: 급변(|r|>th), 볼륨 z-score 스파이크 -> crash_dummy(0~1 감쇠)
- 3) _to_prophet_frame: Prophet 포맷 (ds, y, volume, crash_dummy) 변환
- 4) fit_predict: 모델 생성(회귀자 등록) -> 학습 -> 미래프레임(회귀자 포함) -> 예측 요약 반환

[전략 연동 가이드]
- 매수 예: yhat_avg가 현재가 대비 +a% 이상 && 신뢰구간 폭이 임계 이하일 때만 유효
- 매도 예: yhat_avg가 현재가 대비 -a% 이하 && 신뢰구간 폭이 임계 이하일 때만 유효
- HOLD: 신뢰구간 과도 확대/이벤트 직후 구간 등 불확실성 큰 경우
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Literal, List
import numpy as np
import pandas as pd

# Prophet import 체크
try:
    from prophet import Prophet
except Exception as e:
    Prophet = None
    _IMPORT_ERROR = e

@dataclass
class ProphetModel:
    # ====== 핵심 하이퍼파라미터 ======
    freq: str = "1d"                      # 예측할 주기 (data_process/data_processor.py로 전처리된 .csv 파일 필요)
    horizon: int = 3                      # 예측 스텝 수 (freq 곱하기 horizon)
    start_date: str = "2025-04-01"        # 학습 시작일(UTC 기준)
    end_date: Optional[str] = "2025-10-28T00:09:00+00:00"     #학습 종료일(UTC 기준)
    interval_width: float = 0.80          # 신뢰구간 폭 (0.8~0.95 사이로 설정함)
    changepoint_prior_scale: float = 0.15 # 추세 변곡점 민감도(클수록 민감)
    seasonality_mode: Literal["additive","multiplicative"] = "additive"

    # 급변 감지 기준 (전처리/이벤트 플래그용)
    ret_spike_thresh: float = 0.08   # |수익률| > 8% 이면 급변으로 간주
    vol_z_window: int = 60           # 거래량 z-score 계산 롤링 윈도우(최소 20)
    vol_z_thresh: float = 2.0        # 거래량 z-score > 2.5 이면 스파이크
    crash_decay_bars: int = 1        # 급변 이후 n개 구간까지 감쇠(1, 0.5, 0.25...)

    # 휴일 이벤트 관련
    use_fomc_holidays: bool = True
    holidays_prior_scale: float = 3.0   # 이벤트(휴일) 영향 강도
    pre_event_days: int = 1             # 발표 1 전까지 영향

    # 로그-가격, 매물대, 체감적 하이퍼 튜닝
    use_log_price: bool = True
    n_changepoints: int = 35            # 추세 변곡점 더 촘촘하게
    changepoint_range: float = 0.9     # 최근 구간에도 변곡 허용

    # 매물대(VAP) 관련
    use_vap: bool = False
    vap_window: int = 30                 # 매물대 롤링 윈도우
    vap_bins: int = 60                  # 가격 구간 개수
    vap_smooth_sigma: float = 1.0       # 히스토그램 가우시안 스무딩

    """
    # 공포 탐욕 지수 추가
    use_fgi: bool = False
    fgi_col: str = "fgi"
    fgi_lag_days: int = 1           # 하루 지연
    fgi_prior_scale: float = 0.5    # 영향 강도)
    """

    # ====== 내부 상태(로그/디버깅용) ======
    _last_fit_samples: int = field(init=False, default=0)  # 마지막 학습 샘플 수

    # ---------------------------
    # 0) 환경 점검: Prophet import 성공 여부
    # ---------------------------
    def _check_env(self) -> bool:
        if Prophet is None:
            print(f"[ProphetModel] ImportError: {getattr(_IMPORT_ERROR, 'msg', _IMPORT_ERROR)}")
            return False
        return True

    # ---------------------------
    # 1) 입력 검증 + 최소 정리
    # ---------------------------
    def _validate_input(self, df_in: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        - 타입/필수 컬럼 검사("close","volume")
        - 인덱스 UTC 통일, 정렬/중복 제거
        - 숫자형 강제 변환, NaN/Inf 제거
        - 가격>0, 거래량>=0 필터
        - 학습 기간 필터: start_date부터 end_date까지
        """
        # 1) 타입 검사
        if not isinstance(df_in, pd.DataFrame):
            print("[ProphetModel][validate] 입력은 DataFrame이어야 합니다.")
            return None

        # 2) 필수 컬럼 검사
        required = {"close", "volume"}
        missing = [c for c in required if c not in df_in.columns]
        if missing:
            print(f"[ProphetModel][validate] 필수 컬럼 누락: {missing}")
            return None

        df = df_in.copy()

        # 3) DatetimeIndex 여부
        if not isinstance(df.index, pd.DatetimeIndex):
            print("[ProphetModel][validate] DatetimeIndex가 필요합니다.")
            return None

        # 4) UTC 통일
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        # 5) 정렬 + 중복 제거
        df = df[~df.index.duplicated(keep="last")].sort_index()

        # 6) 숫자형 강제 + NaN/Inf 제거
        for c in required:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=list(required), inplace=True)

        # 7) 음수/영값 필터
        df = df[(df["close"] > 0) & (df["volume"] >= 0)]

        # 8) 학습 기간 필터
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

        # 9) 최소 샘플 안전장치 (weekly seasonality 학습 위해 어느 정도 길이 필요)
        if len(df) < 200:  # 너무 짧으면 경고
            print(f"[ProphetModel][validate] 샘플이 적을 수 있음: len={len(df)}")
        return df

    # ---------------------------
    # 2) 급변/볼륨스파이크 감지 → crash_dummy 생성
    # ---------------------------
    def _build_event_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        - |r_t| > ret_spike_thresh -> 급변 플래그
        - 거래량 z-score > vol_z_thresh -> 볼륨 스파이크 플래그
        - 두 조건 중 하나라도 TRUE면 1, 그 외 0
        - crash_decay_bars만큼 이후 캔들에 감쇠 적용
        """
        out = df.copy()

        # 수익률(전 스텝 대비)
        ret = out["close"].pct_change().fillna(0.0)
        ret_flag = (ret.abs() > self.ret_spike_thresh).astype(int)

        # 거래량 z-score
        w = max(self.vol_z_window, 20)
        vol = out["volume"].astype(float)
        vol_mean = vol.rolling(w, min_periods=20).mean()
        vol_std = vol.rolling(w, min_periods=20).std()
        den = vol_std.replace(0, np.nan)
        vol_z = (vol - vol_mean) / den
        vol_z = vol_z.fillna(0.0)
        vol_flag = (vol_z > self.vol_z_thresh).astype(int)

        # 기본 플래그
        base = ((ret_flag == 1) | (vol_flag == 1)).astype(int).values

        # 감쇠 적용
        decay = np.zeros_like(base, dtype=float)
        for i in range(len(base)):
            if base[i] == 1:
                decay[i] += 1.0
                for k in range(1, self.crash_decay_bars + 1):
                    j = i + k
                    if j < len(decay):
                        decay[j] = max(decay[j], 1.0 / (2 ** k))

        out["crash_dummy"] = decay  # 0~1 범위 이벤트 강도
        return out

    # ---------------------------
    # 3) Prophet 입력 포맷 변환
    # ---------------------------
    def _to_prophet_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        - Prophet 포맷: ds(시간), y(값), volume(보조회귀), crash_dummy(이벤트 강도)
        - Prophet은 tz-aware datetime을 받지 않으므로 tz 제거(naive)
        """
        flg = self._build_event_flags(df)
        if self.use_vap:
            flg = self._build_vap_features(flg)

        ds = flg.index.tz_convert(None)
        pdf = pd.DataFrame(
            {
                "ds": ds,
                "y": np.log(flg["close"].values) if self.use_log_price else flg["close"].values,
                "volume": flg["volume"].values,
                "crash_dummy": flg["crash_dummy"].values,

            }
        )
        if self.use_vap:
            pdf["vap_poc_diff"] = flg["vap_poc_diff"].values
            pdf["vap_val_dist"] = flg["vap_val_dist"].values
            pdf["vap_vah_dist"] = flg["vap_vah_dist"].values



        """
        pdf = pd.DataFrame(
            {
                "ds": ds,
                "y": flg["close"].values,
                "volume": flg["volume"].values,
                "crash_dummy": flg["crash_dummy"].values,

            }
        
        """
        if pdf[["ds", "y", "volume", "crash_dummy"]].isna().any().any():
            print("[ProphetModel][transform] pdf에 NaN 존재 → 중단")
            return pd.DataFrame()
        return pdf
    # ---------------------------
    # 4) 휴일 효과 추가 (현재: FOMC 금리 발표
    # ---------------------------
    def _build_holidays_df(self) -> pd.DataFrame:
        """
        2025년 FOMC 정례회의 ‘결정일(둘째 날)’을 이벤트로 사용.
        - 일봉(freq='1d') 기준: 발표 ‘전날’까지도 영향 주려면 lower_window = -self.pre_event_days
        - upper_window = 0 (발표 당일)
        """
        dates_2025 = [
            "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
            "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
        ]
        ds = pd.to_datetime(dates_2025, utc=True)
        holidays_df = pd.DataFrame({
            "holiday": "FOMC",
            "ds": ds.tz_convert(None),           # Prophet은 naive datetime 권장
            "lower_window": -int(self.pre_event_days),
            "upper_window": 0
        })
        # 학습 기간(start_date~end_date)에 맞춰 잘라주기(옵션)
        start_ts = pd.Timestamp(self.start_date, tz="UTC")
        end_ts = pd.Timestamp(self.end_date, tz="UTC") if self.end_date else None
        mask = (ds >= start_ts) & ((end_ts is None) | (ds <= end_ts))
        holidays_df = holidays_df.loc[mask]
        return holidays_df.reset_index(drop=True)

    def _build_vap_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        최근 vap_window개 구간으로 VAP(가격별 거래량) 계산 → POC/VAL/VAH 추정 →
        현 종가 대비 상대거리 피처 3개 생성.
        """
        if not self.use_vap or len(df) < max(self.vap_window, 30):
            out = df.copy()
            out["vap_poc_diff"] = 0.0
            out["vap_val_dist"]  = 0.0
            out["vap_vah_dist"]  = 0.0
            return out

        import numpy as np
        try:
            from scipy.ndimage import gaussian_filter1d
        except Exception:
            gaussian_filter1d = None

        out = df.copy()
        tail = out.iloc[-self.vap_window:]
        prices  = tail["close"].astype(float).values
        volumes = tail["volume"].astype(float).values

        # 가격 구간(bin) 생성
        p_min, p_max = prices.min(), prices.max()
        if p_min == p_max:  # 모든 값이 동일한 극단 케이스
            out["vap_poc_diff"] = 0.0
            out["vap_val_dist"]  = 0.0
            out["vap_vah_dist"]  = 0.0
            return out

        bin_edges = np.linspace(p_min, p_max, num=self.vap_bins + 1)
        hist, edges = np.histogram(prices, bins=bin_edges, weights=volumes)

        # 가우시안 스무딩(옵션)
        if self.vap_smooth_sigma and self.vap_smooth_sigma > 0 and gaussian_filter1d is not None:
            hist = gaussian_filter1d(hist.astype(float), sigma=self.vap_smooth_sigma)

        # POC(최대 거래량), VAH/VAL(상위/하위 누적 15% 예시)
        bin_centers = (edges[:-1] + edges[1:]) / 2.0
        poc_price = float(bin_centers[np.argmax(hist)])

        cum = hist.cumsum()
        total = cum[-1] if cum[-1] > 0 else 1.0
        lower_cut = total * 0.15
        upper_cut = total * 0.85
        # VAL = 누적이 lower_cut을 처음 넘는 지점, VAH = 누적이 upper_cut을 처음 넘는 지점
        val_price = float(bin_centers[np.searchsorted(cum, lower_cut, side="left")])
        vah_price = float(bin_centers[np.searchsorted(cum, upper_cut, side="left")])

        # 마지막 종가 기준 상대 거리
        last_close = float(out["close"].iloc[-1])
        out["vap_poc_diff"] = (last_close - poc_price) / last_close
        out["vap_val_dist"]  = (last_close - val_price) / last_close
        out["vap_vah_dist"]  = (last_close - vah_price) / last_close

        # 전체 구간에 “현재 구조”를 반복 적용(Prophet 회귀자는 시계열 값 필요.
        # 여기서는 ‘마지막 상태를 유지’하는 보수적 가정)
        out[["vap_poc_diff","vap_val_dist","vap_vah_dist"]] = out[["vap_poc_diff","vap_val_dist","vap_vah_dist"]].ffill().bfill()

        return out

    # ---------------------------
    # 5) 학습 + 멀티스텝 예측(horizon>1)
    # ---------------------------
    def fit_predict(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        - 전체 파이프라인:
          _check_env -> _validate_input -> _to_prophet_frame
          모델 생성(add_regressor 등록: volume, crash_dummy)
          fit -> 미래 프레임(회귀자 포함) -> 예측 -> horizon 요약 반환
        - 반환:
          {"yhat": 평균, "yhat_lower": 최소, "yhat_upper": 최대}
        """
        # 5-0) 환경
        if not self._check_env():
            return None

        # 5-1) 입력 검증
        dfv = self._validate_input(df)
        if dfv is None or len(dfv) == 0:
            return None

        self._last_fit_samples = len(dfv)

        # 5-2) Prophet 입력
        pdf = self._to_prophet_frame(dfv)
        if pdf.empty:
            return None

        holidays_df = None
        if self.use_fomc_holidays:
            holidays_df = self._build_holidays_df()

        # 5-3) 모델 생성
        m = Prophet(
            interval_width=self.interval_width,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_mode=self.seasonality_mode,
            daily_seasonality=False,   # 24h 주기 비활성
            weekly_seasonality=True,  # 주간 주기 허용
            yearly_seasonality=True,  # 중기 신호용, 연간 주기 허용
            n_changepoints=self.n_changepoints,
            changepoint_range=self.changepoint_range,
            seasonality_prior_scale=1.0,
            holidays=holidays_df,
            holidays_prior_scale=self.holidays_prior_scale
        )

        m.add_regressor("volume", standardize=True, prior_scale=0.05)
        m.add_regressor("crash_dummy", standardize=True, prior_scale=0.3)

        if self.use_vap:
            m.add_regressor("vap_poc_diff", standardize=True, prior_scale=0.2)
            m.add_regressor("vap_val_dist", standardize=True, prior_scale=0.1)
            m.add_regressor("vap_vah_dist", standardize=True, prior_scale=0.1)

        # 5-4) 학습
        try:
            m.fit(pdf)
        except Exception as e:
            print(f"[ProphetModel][fit] 실패: {e}")
            return None

        # 5-5) 예측
        try:
            future = m.make_future_dataframe(
                periods=self.horizon,
                freq=self.freq,
                include_history=False
            )
            # 미래 회귀자 채움
            # - volume: 최근 14일 평균
            # - crash_dummy: 0(미래 이벤트 미지수, 정상상태 가정)
            k = min(14, len(dfv))
            hist_vol = dfv["volume"].iloc[-k:]
            p10, p90 = hist_vol.quantile([0.10, 0.90])
            vol_med = float(hist_vol.median())
            future["volume"] = float(np.clip(vol_med, p10, p90))

            # 미래 이벤트는 미지수 = 0
            future["crash_dummy"] = 0.0

            # VAP: 미래도 현재 구조 유지 가정 -> 마지막 값으로 채훔
            if self.use_vap:
                future["vap_poc_diff"] = float(pdf["vap_poc_diff"].iloc[-1])
                future["vap_val_dist"] = float(pdf["vap_val_dist"].iloc[-1])
                future["vap_vah_dist"] = float(pdf["vap_vah_dist"].iloc[-1])

            fcst = m.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]

            if self.use_log_price:
                # yhat 계열 역변환
                fcst["yhat"] = np.exp(fcst["yhat"])
                fcst["yhat_lower"] = np.exp(fcst["yhat_lower"])
                fcst["yhat_upper"] = np.exp(fcst["yhat_upper"])

            # 예측 감쇠: 차이가커질수록 예측 변화 완롸
            last_close = float(dfv["close"].iloc[-1])
            k = 4.0
            def _soft_attenuate(fcst: pd.DataFrame, last_close: float, k: float = 4.0) -> pd.DataFrame:
                out = fcst.copy()
                delta = out["yhat"] - last_close
                delta_pct = delta / last_close

                # 감쇠 계수
                weight = 1.0 / (1.0 + k * np.abs(delta_pct))

                # 중심값 조정
                out["yhat"] = last_close + delta * weight

                # 신뢰구간도 동일한 weight로 조정
                low_delta = out["yhat_lower"] - last_close
                up_delta = out["yhat_upper"] - last_close
                out["yhat_lower"] = last_close + low_delta * weight
                out["yhat_upper"] = last_close + up_delta * weight

                # 혹시 상하 역전 방지
                lower = out[["yhat_lower", "yhat_upper"]].min(axis=1)
                upper = out[["yhat_lower", "yhat_upper"]].max(axis=1)
                out["yhat_lower"] = lower
                out["yhat_upper"] = upper

                return out
            fcst = _soft_attenuate(fcst, last_close, k=4.0)

            # horizon 요약(전략 연동용 대표값)
            yhat_avg = float(fcst["yhat"].mean())                # 중심값: 평균
            yhat_low = float(fcst["yhat_lower"].min())           # 보수적 하단
            yhat_up  = float(fcst["yhat_upper"].max())           # 보수적 상단

            result = {
                "yhat": yhat_avg,
                "yhat_lower": yhat_low,
                "yhat_upper": yhat_up
            }
            return result

        except Exception as e:
            print(f"[ProphetModel][predict] 실패: {e}")
            return None

