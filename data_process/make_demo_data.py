
import pandas as pd
import sys
import os

from pathlib import Path

# (같은 폴더에 있으므로 sys.path.append는 필요 없습니다)

# 같은 폴더에 있는 모듈들 바로 불러오기
from data_loader import load_raw_csv
from data_processor import resample_ohlcv

def create_demo_dataset_15m():
    print("--- [데모용 15분봉 데이터 생성 시작] ---")

    # 1. 설정
    start_date = "2025-09-01 00:00:00"
    end_date = "2025-10-31 23:59:59"

    # ★★★ 수정된 부분 (경로) ★★★
    # 이 파일이 있는 폴더 (data_process)
    current_dir = Path(__file__).resolve().parent
    # 프로젝트 루트 폴더 (한 단계 위로 올라감)
    project_root = current_dir.parent

    raw_csv_path = project_root / "data/raw/binance_BTCUSDT_1m_raw.csv"
    output_csv_path = project_root / "data/processed/demo_BTCUSDT_15m.csv"

    # 2. 원본 1분봉 데이터 불러오기
    if not raw_csv_path.exists():
        print(f"❌ 오류: 원본 파일이 없습니다. ({raw_csv_path})")
        print("먼저 data_loader.py를 실행해서 1분봉 데이터를 받아주세요.")
        return

    print(f"1. 원본 데이터 로딩 중... ({raw_csv_path})")
    df = load_raw_csv(raw_csv_path)

    # 3. 날짜 자르기
    print(f"2. 날짜 필터링 중... ({start_date} ~ {end_date})")
    filtered_df = df.loc[start_date:end_date].copy()

    if filtered_df.empty:
        print("❌ 오류: 해당 기간의 데이터가 없습니다.")
        return

    print(f"   -> 추출된 데이터: {len(filtered_df)}개 행")

    # 4. 15분봉으로 변환
    print("3. 15분봉으로 변환(Resampling) 중...")
    df_15m = resample_ohlcv(filtered_df, out_freq="15min")

    # 5. CSV 파일로 저장
    print(f"4. 파일 저장 중... ({output_csv_path})")
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_15m.to_csv(output_csv_path, index=True)

    print("\n✅ [성공] 데모용 15분봉 데이터 생성 완료!")
    print(f"   - 저장 경로: {output_csv_path}")
    print(f"   - 데이터 개수: {len(df_15m)}개")
    print(f"   - 시작: {df_15m.index[0]}")
    print(f"   - 끝: {df_15m.index[-1]}")

if __name__ == "__main__":
    create_demo_dataset_15m()