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
from typing import Optional, Dict, Literal
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
    start_date: str = "2025-01-01"        # 학습 시작일(UTC 기준)
    end_date: Optional[str] = "2025-10-12T00:09:00+00:00"     #학습 종료일(UTC 기준)
    interval_width: float = 0.80          # 신뢰구간 폭 (0.8~0.95 사이로 설정함)
    changepoint_prior_scale: float = 0.20 # 추세 변곡점 민감도(클수록 민감)
    seasonality_mode: Literal["additive","multiplicative"] = "additive"

    # 급변 감지 기준 (전처리/이벤트 플래그용)
    ret_spike_thresh: float = 0.08   # |수익률| > 8% 이면 급변으로 간주
    vol_z_window: int = 100          # 거래량 z-score 계산 롤링 윈도우(최소 20)
    vol_z_thresh: float = 3.0        # 거래량 z-score > 3 이면 스파이크
    crash_decay_bars: int = 3        # 급변 이후 n개 구간까지 감쇠(1, 0.5, 0.25...)

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
        ds = flg.index.tz_convert(None)
        pdf = pd.DataFrame(
            {
                "ds": ds,
                "y": flg["close"].values,
                "volume": flg["volume"].values,
                "crash_dummy": flg["crash_dummy"].values,
            }
        )
        if pdf[["ds", "y", "volume", "crash_dummy"]].isna().any().any():
            print("[ProphetModel][transform] pdf에 NaN 존재 → 중단")
            return pd.DataFrame()
        return pdf

    # ---------------------------
    # 4) 학습 + 멀티스텝 예측(horizon>1)
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
        # 4-0) 환경
        if not self._check_env():
            return None

        # 4-1) 입력 검증
        dfv = self._validate_input(df)
        if dfv is None or len(dfv) == 0:
            return None

        self._last_fit_samples = len(dfv)

        # 4-2) Prophet 입력
        pdf = self._to_prophet_frame(dfv)
        if pdf.empty:
            return None

        # 4-3) 모델 생성
        m = Prophet(
            interval_width=self.interval_width,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_mode=self.seasonality_mode,
            daily_seasonality=False,   # 24h 주기 비활성
            weekly_seasonality=True,  # 주간 주기 허용
            yearly_seasonality=True,  # 중기 신호용, 연간 주기 허용
            n_changepoints=10,
            seasonality_prior_scale=5.0
        )
        m.add_regressor("volume", standardize=True, prior_scale=1.0)
        m.add_regressor("crash_dummy", standardize=True, prior_scale=0.5)

        # 4-4) 학습
        try:
            m.fit(pdf)
        except Exception as e:
            print(f"[ProphetModel][fit] 실패: {e}")
            return None

        # 4-5) 예측
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


            fcst = m.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]

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

