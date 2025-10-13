# data_fetcher.py (최종 완성본)

import pandas as pd
from binance.client import Client
from binance import ThreadedWebsocketManager # 실시간 데이터를 위한 Websocket 매니저
import logging
import time
import os
from typing import Callable, List, Tuple, Union


# --- 기본 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. 과거 데이터 수집 함수 ---
def fetch_klines_binance(symbol: str, interval: str, start_str: str, end_str: str = None) -> pd.DataFrame:
    # 이 함수는 바이낸스 클라이언트를 생성합니다. (공개 데이터는 API 키 필요 없음)
    client = Client()
    # "어떤 코인을, 언제부터 언제까지 가져오기 시작합니다" 라는 정보를 로그로 남깁니다.
    logging.info(f"Fetching klines for {symbol} from {start_str} to {end_str}.")

    try:
        # get_historical_klines 함수는 데이터가 많아도 알아서 여러 번 나눠서 요청하고 모든 데이터를 합쳐줍니다.
        # end_str이 None이면 최신까지 가져오기 (end_str 파라미터 생략)
        if end_str is None or end_str == "now UTC":
            klines = client.get_historical_klines(symbol, interval, start_str)
        else:
            klines = client.get_historical_klines(symbol, interval, start_str, end_str)

        # 받아온 데이터가 비어있다면 경고 로그를 남기고 빈 DataFrame을 반환합니다.
        if not klines:
            logging.warning("No data found for the specified period.")
            return pd.DataFrame()

        # 받아온 데이터를 Pandas DataFrame(표)으로 가공합니다.
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])

        # 필요한 6개 열만 선택합니다.
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        # 'timestamp' 열의 데이터(ms 숫자)를 UTC 기준 날짜/시간 형식으로 변환합니다.
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        # 'timestamp' 열을 DataFrame의 인덱스로 설정합니다.
        df.set_index('timestamp', inplace=True)
        # 모든 숫자 데이터의 타입을 소수점이 있는 숫자(float)로 통일합니다.
        df = df.astype(float)
        # 시간순으로 정렬하고 중복 데이터를 제거합니다.
        df = df[~df.index.duplicated(keep='first')]
        df.sort_index(inplace=True)

        # 성공 로그를 남기고 최종 DataFrame을 반환합니다.
        logging.info(f"Successfully fetched {len(df)} klines for {symbol}.")
        return df

    except Exception as e:
        # 에러 발생 시 로그를 남기고 프로그램을 중단시킵니다.
        logging.error(f"An error occurred while fetching data: {e}")
        raise

# --- 2. 실시간 데이터 수신 함수 ---
def stream_klines_binance(symbol: str, interval: str, on_bar: Callable[[dict], None]):
    # 실시간 데이터를 관리하는 ThreadedWebsocketManager 객체를 생성합니다.
    twm = ThreadedWebsocketManager()
    # 매니저를 시작합니다. (백그라운드에서 실행)
    twm.start()

    # 실시간 메시지를 받았을 때 처리할 내부 함수를 정의합니다.
    def handle_socket_message(msg):
        # 메시지에서 캔들 데이터('k')를 추출합니다.
        kline = msg['k']
        # 캔들이 완성되었는지('x' 키 값이 True인지) 확인합니다.
        if kline['x']:
            # 요구사항에 맞게 데이터를 정리합니다.
            bar_data = {
                "timestamp": kline['t'],
                "open": float(kline['o']),
                "high": float(kline['h']),
                "low": float(kline['l']),
                "close": float(kline['c']),
                "volume": float(kline['v'])
            }
            # 사용자가 전달한 on_bar 함수를 호출하여 정리된 데이터를 전달합니다.
            on_bar(bar_data)

    # 지정된 심볼과 간격으로 kline 소켓(실시간 스트림)을 시작합니다.
    # 메시지가 도착하면 위에서 만든 handle_socket_message 함수가 실행됩니다.
    stream_name = twm.start_kline_socket(callback=handle_socket_message, symbol=symbol, interval=interval)

    # WebSocket이 계속 실행되도록 메인 스레드를 대기시킵니다. (Ctrl+C로 종료 가능)
    # 이 라이브러리는 연결이 끊겼을 때 자동으로 재연결을 시도합니다.
    print(f"Starting real-time kline stream for {symbol}. Press Ctrl+C to stop.")
    twm.join()


# --- 3. 누락 데이터 채우기 함수 ---
def backfill_binance(symbol: str, interval: str, gaps: List[Tuple[int, int]]) -> pd.DataFrame:
    # 모든 누락 구간에서 가져온 DataFrame들을 담을 빈 리스트를 만듭니다.
    all_dfs = []
    # "몇 개의 누락 구간에 대한 데이터 채우기를 시작합니다" 라는 로그를 남깁니다.
    logging.info(f"Starting backfill for {len(gaps)} gaps for {symbol}.")

    # 입력받은 누락 구간 리스트(gaps)를 하나씩 순회합니다.
    for i, (start_ms, end_ms) in enumerate(gaps):
        # 각 구간의 시작과 끝 시간을 문자열로 변환합니다.
        start_str = pd.to_datetime(start_ms, unit='ms', utc=True).isoformat()
        end_str = pd.to_datetime(end_ms, unit='ms', utc=True).isoformat()

        # "몇 번째 누락 구간을 채우고 있습니다" 라는 로그를 남깁니다.
        logging.info(f"Backfilling gap {i+1}/{len(gaps)} from {start_str} to {end_str}...")
        try:
            # 1번 함수(fetch_klines_binance)를 재사용하여 현재 구간의 데이터를 가져옵니다.
            df_gap = fetch_klines_binance(symbol, interval, start_str, end_str)
            # 만약 해당 구간의 데이터가 비어있지 않다면,
            if not df_gap.empty:
                # 결과 리스트(all_dfs)에 추가합니다.
                all_dfs.append(df_gap)
        except Exception as e:
            # 에러가 나면 해당 구간은 건너뜁니다.
            logging.error(f"Failed to backfill gap from {start_str} to {end_str}: {e}")

    # 만약 모든 구간을 확인했는데도 채워진 데이터가 하나도 없다면,
    if not all_dfs:
        logging.warning("Backfill resulted in no new data.")
        return pd.DataFrame()

    # 각 구간별로 가져온 모든 DataFrame 조각들을 하나의 큰 DataFrame으로 합칩니다.
    combined_df = pd.concat(all_dfs)
    # 합친 후 혹시 모를 중복 데이터를 제거합니다.
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
    # 시간순으로 다시 정렬합니다.
    combined_df.sort_index(inplace=True)

    # "데이터 채우기 완료. 총 몇 개의 데이터를 가져왔습니다" 라는 로그를 남깁니다.
    logging.info(f"Backfill complete. Total {len(combined_df)} klines fetched.")
    # 최종적으로 합쳐지고 정리된 DataFrame을 반환합니다.
    return combined_df

# --- 예제 코드 실행 부분 ---
# 이 파일을 직접 실행했을 때만 아래 코드가 동작합니다.
if __name__ == '__main__':
    # --- 1. 과거 데이터 수집 테스트 ---
    print("--- 1. Testing fetch_klines_binance ---")

    try:
        # 과거 데이터 수집 함수를 호출합니다.
        historical_data = fetch_klines_binance(
            symbol="BTCUSDT",
            interval=Client.KLINE_INTERVAL_3MINUTE,
            start_str="2025-01-01",
            end_str=None
        )
        print(f"✅ Success! Fetched {len(historical_data)} rows")
        print("Data preview:")
        print(historical_data.head())

    except Exception as e:
        print(f"❌ An error occurred: {e}")

    print("\n" + "="*50 + "\n")

    # --- 3. 누락된 구간 데이터 채우기 테스트 ---
    print("--- 3. Testing backfill_binance ---")
    try:
        now_ms = int(time.time() * 1000)
        day_ms = 24 * 60 * 60 * 1000
        gap1_start = now_ms - 10 * day_ms # 10일 전
        gap1_end = gap1_start + 3 * 60 * 60 * 1000 # 3시간 분량
        gap2_start = now_ms - 5 * day_ms # 5일 전
        gap2_end = gap2_start + 2 * 60 * 60 * 1000 # 2시간 분량

        missing_gaps = [(gap1_start, gap1_end), (gap2_start, gap2_end)]
        backfilled_data = backfill_binance("BTCUSDT", Client.KLINE_INTERVAL_15MINUTE, missing_gaps)
        print("✅ Backfill test complete. Fetched data:")
        print(backfilled_data)
    except Exception as e:
        print(f"❌ An error occurred: {e}")

    print("\n" + "="*50 + "\n")

    # --- 2. 실시간 데이터 수신 테스트 (기본적으로 주석 처리) ---
    print("--- 2. Testing stream_klines_binance ---")

    # 실시간 데이터가 도착했을 때 화면에 출력할 함수를 정의합니다.
    def my_bar_handler(bar_data):
        bar_time = pd.to_datetime(bar_data['timestamp'], unit='ms', utc=True)
        print(f"  -> New confirmed bar: Time={bar_time}, Close={bar_data['close']}")

    # 아래 실시간으로 종가 받아오는 코드 #지우고 사용 ㄱ

    # try:
    #     stream_klines_binance(
    #         symbol="BTCUSDT",
    #         interval=Client.KLINE_INTERVAL_1MINUTE,
    #         on_bar=my_bar_handler
    #     )
    # except KeyboardInterrupt:
    #     print("\nStream stopped by user.")
    # except Exception as e:
    #     print(f"❌ An error occurred during stream test: {e}")