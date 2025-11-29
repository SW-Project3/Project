"""
Flask 웹 서버 - 트레이딩 차트 데이터 제공
"""
from flask import Flask, jsonify, send_from_directory
try:
    from flask_cors import CORS
except ImportError:
    # flask_cors가 없으면 CORS 없이 실행
    CORS = None
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# 프로젝트 루트를 sys.path에 추가
# 도커 환경에서는 /app이 루트
if os.path.exists('/app'):
    project_root = Path('/app')
else:
    # 로컬 개발 환경
    project_root = Path(__file__).resolve().parent.parent.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from main import indicators
from main.indicator_strategy_connector import generate_signals_indicator_only, DEFAULT_PARAMS

app = Flask(__name__)
if CORS:
    CORS(app)
else:
    # CORS 헤더 수동 추가
    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response

# 전역 변수로 데이터 캐싱
cached_data = None
cached_signals = None


def load_and_prepare_data(csv_path: str, start_date: str = "2025-09-01", end_date: str = "2025-10-31"):
    """CSV 파일을 로드하고 날짜 필터링"""
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    
    # 컬럼명 소문자로 변환
    df.columns = [col.lower() for col in df.columns]
    
    # 날짜 필터링
    df = df.loc[start_date:end_date].copy()
    
    # 인덱스가 타임존 정보가 있으면 UTC로 변환
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')
    
    # 컬럼명을 대문자로 변환 (지표 계산 함수가 대문자 컬럼을 기대)
    df = df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    
    return df


def calculate_indicators(df: pd.DataFrame, params: dict = None):
    """지표 계산"""
    if params is None:
        params = DEFAULT_PARAMS
    
    df = df.copy()
    
    # RSI
    df['RSI'] = indicators.rsi(df['Close'], period=params['rsi_period'])
    
    # MACD
    macd_line, signal_line, hist = indicators.macd(
        df['Close'],
        short=params['macd_short'],
        long=params['macd_long'],
        signal=params['macd_signal']
    )
    df['MACD'] = macd_line
    df['MACD_Signal'] = signal_line
    df['MACD_Hist'] = hist
    
    # Bollinger Bands
    ma, upper, lower = indicators.bollinger_bands(
        df['Close'],
        period=params['bb_period'],
        k=params['bb_k']
    )
    df['BB_MA'] = ma
    df['BB_Upper'] = upper
    df['BB_Lower'] = lower
    
    # ATR (손절가/익절가 계산용)
    df['ATR'] = indicators.atr(
        df['High'],
        df['Low'],
        df['Close'],
        period=params['atr_period']
    )
    
    return df


def calculate_asset_curve(df: pd.DataFrame, signals: pd.DataFrame, initial_cash: float = 1000.0):
    """자산 변동 곡선 계산 (간단한 백테스트)"""
    cash = initial_cash
    position = 0.0  # 포지션 수량 (양수: 롱, 음수: 숏)
    entry_price = None
    stop_loss = None
    take_profit = None
    
    asset_values = []
    timestamps = []
    
    for ts in df.index:
        price = float(df.at[ts, 'Close'])
        sig = int(signals.at[ts, 'signal']) if ts in signals.index else 0
        
        # 포지션이 있을 때 손절/익절 조건 체크 (신호 체크 전에 먼저 실행)
        if position != 0.0:
            if position > 0:  # 롱 포지션
                if price <= stop_loss or price >= take_profit:
                    # 롱 포지션 청산: 포지션을 현금으로 전환
                    cash = position * price
                    position = 0.0
                    entry_price = None
                    stop_loss = None
                    take_profit = None
            else:  # 숏 포지션
                if price >= stop_loss or price <= take_profit:
                    # 숏 포지션 청산
                    # 진입 시: cash는 그대로, position만 음수로 설정
                    # 청산 시: cash + (진입가 - 현재가) * 포지션 크기
                    pnl = (entry_price - price) * abs(position)
                    cash = cash + pnl
                    position = 0.0
                    entry_price = None
                    stop_loss = None
                    take_profit = None
        
        # 새 신호 발생 시 진입 (포지션이 없을 때만)
        if sig != 0 and position == 0.0:
            if ts not in signals.index:
                continue
                
            entry_price = price
            stop_loss_val = signals.at[ts, 'stop_loss']
            take_profit_val = signals.at[ts, 'take_profit']
            
            if pd.isna(stop_loss_val) or pd.isna(take_profit_val):
                continue
                
            stop_loss = float(stop_loss_val)
            take_profit = float(take_profit_val)
            
            # 전체 잔고로 매수/매도
            if sig > 0:  # 매수 (롱)
                if cash > 0 and entry_price > 0:
                    position = cash / entry_price
                    cash = 0.0
            else:  # 매도 (숏)
                if cash > 0 and entry_price > 0:
                    # 숏 포지션: 현금은 유지하고 포지션만 음수로 설정
                    position = -(cash / entry_price)
                    # cash는 그대로 유지 (청산 시 계산)
        
        # 현재 자산 가치 계산
        if position != 0.0:
            if position > 0:  # 롱 포지션
                asset_value = position * price
            else:  # 숏 포지션
                # 숏 포지션: 현금 + (진입가 - 현재가) * 포지션 크기
                pnl = (entry_price - price) * abs(position)
                asset_value = cash + pnl
        else:
            asset_value = cash
        
        # 자산 가치가 0보다 작아지지 않도록 보정
        asset_value = max(0.0, asset_value)
        
        asset_values.append(asset_value)
        timestamps.append(ts.isoformat())
    
    final_value = asset_values[-1] if asset_values else initial_cash
    return_pct = ((final_value - initial_cash) / initial_cash * 100) if initial_cash > 0 else 0.0
    
    return {
        'timestamp': timestamps,
        'asset_value': asset_values,
        'initial_cash': initial_cash,
        'final_value': final_value,
        'return_pct': return_pct
    }


def load_data():
    """데이터 로드 및 캐싱"""
    global cached_data, cached_signals
    
    if cached_data is not None:
        return cached_data, cached_signals
    
    # 도커 환경에서는 볼륨 마운트된 경로 사용
    # 볼륨 마운트: ../data:/app/data:ro
    if os.path.exists('/app/data/processed/demo_BTCUSDT_15m.csv'):
        csv_path = Path('/app/data/processed/demo_BTCUSDT_15m.csv')
    elif os.path.exists('/app/data/processed'):
        # 데이터 디렉토리는 있지만 파일이 없는 경우
        csv_path = project_root / "data" / "processed" / "demo_BTCUSDT_15m.csv"
    else:
        csv_path = project_root / "data" / "processed" / "demo_BTCUSDT_15m.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {csv_path}")
    
    # 데이터 로드
    df = load_and_prepare_data(str(csv_path))
    
    # 지표 계산
    df = calculate_indicators(df)
    
    # 신호 생성
    signals = generate_signals_indicator_only(df)
    
    # 캐싱
    cached_data = df
    cached_signals = signals
    
    return df, signals


@app.route('/chart-data')
@app.route('/api/chart-data')
def get_chart_data():
    """차트 데이터 API"""
    try:
        from flask import request
        
        # 날짜 범위 파라미터 가져오기
        start_date = request.args.get('start_date', '2025-09-01')
        end_date = request.args.get('end_date', '2025-10-31')
        
        print(f"API 호출 받음: /api/chart-data (기간: {start_date} ~ {end_date})")
        
        # 전체 데이터 로드
        df_full, signals_full = load_data()
        
        # 날짜 필터링
        if start_date == '2025-09-01' and end_date == '2025-10-31':
            # 전체 기간이면 필터링하지 않음
            df = df_full.copy()
            signals = signals_full.copy()
            print("전체 기간 데이터 사용")
        else:
            # 선택된 기간으로 필터링
            try:
                # 날짜 문자열을 datetime으로 변환
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                # 시간까지 포함 (하루 끝까지)
                end_dt = end_dt + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                
                # 타임존 정보가 있으면 UTC로 변환
                if df_full.index.tz is not None:
                    start_dt = start_dt.tz_localize('UTC') if start_dt.tz is None else start_dt.tz_convert('UTC')
                    end_dt = end_dt.tz_localize('UTC') if end_dt.tz is None else end_dt.tz_convert('UTC')
                
                df = df_full.loc[start_dt:end_dt].copy()
                # 신호도 해당 기간으로 필터링
                signals = signals_full.loc[signals_full.index.isin(df.index)].copy()
                print(f"필터링: {start_date} ~ {end_date}")
            except Exception as e:
                print(f"날짜 필터링 오류: {e}")
                df = df_full.copy()
                signals = signals_full.copy()
        
        print(f"필터링 후 데이터: {len(df)} 행, {len(signals)} 신호")
        
        # 데이터를 JSON으로 변환
        timestamps = [ts.isoformat() for ts in df.index]
        
        # 가격 데이터
        price_data = {
            'timestamp': timestamps,
            'open': df['Open'].fillna(0).tolist(),
            'high': df['High'].fillna(0).tolist(),
            'low': df['Low'].fillna(0).tolist(),
            'close': df['Close'].fillna(0).tolist(),
            'volume': df['Volume'].fillna(0).tolist()
        }
        
        # 볼린저 밴드
        bb_data = {
            'timestamp': timestamps,
            'upper': df['BB_Upper'].fillna(0).tolist(),
            'middle': df['BB_MA'].fillna(0).tolist(),
            'lower': df['BB_Lower'].fillna(0).tolist()
        }
        
        # RSI
        rsi_data = {
            'timestamp': timestamps,
            'rsi': df['RSI'].fillna(50).tolist()
        }
        
        # MACD
        macd_data = {
            'timestamp': timestamps,
            'macd': df['MACD'].fillna(0).tolist(),
            'signal': df['MACD_Signal'].fillna(0).tolist(),
            'hist': df['MACD_Hist'].fillna(0).tolist()
        }
        
        # 매수/매도 신호
        buy_signals = signals[signals['signal'] == 1]
        sell_signals = signals[signals['signal'] == -1]
        
        buy_positions = []
        for idx in buy_signals.index:
            if idx in df.index:
                buy_positions.append({
                    'timestamp': idx.isoformat(),
                    'price': float(df.at[idx, 'Close']),
                    'entry_price': float(buy_signals.at[idx, 'entry_price']),
                    'stop_loss': float(buy_signals.at[idx, 'stop_loss']),
                    'take_profit': float(buy_signals.at[idx, 'take_profit'])
                })
        
        sell_positions = []
        for idx in sell_signals.index:
            if idx in df.index:
                sell_positions.append({
                    'timestamp': idx.isoformat(),
                    'price': float(df.at[idx, 'Close']),
                    'entry_price': float(sell_signals.at[idx, 'entry_price']),
                    'stop_loss': float(sell_signals.at[idx, 'stop_loss']),
                    'take_profit': float(sell_signals.at[idx, 'take_profit'])
                })
        
        # 자산 변동 계산 (백테스트)
        asset_data = calculate_asset_curve(df, signals, initial_cash=1000.0)
        
        return jsonify({
            'success': True,
            'price': price_data,
            'bollinger_bands': bb_data,
            'rsi': rsi_data,
            'macd': macd_data,
            'buy_positions': buy_positions,
            'sell_positions': sell_positions,
            'asset_curve': asset_data
        })
    
    except Exception as e:
        import traceback
        error_msg = str(e)
        error_trace = traceback.format_exc()
        print(f"API 오류 발생: {error_msg}")
        print(f"트레이스백:\n{error_trace}")
        return jsonify({
            'success': False,
            'error': error_msg,
            'traceback': error_trace
        }), 500


@app.route('/health')
def health():
    """헬스 체크"""
    return jsonify({'status': 'ok', 'service': 'flask-chart-api'})

@app.route('/api/health')
def api_health():
    """API 헬스 체크"""
    return jsonify({'status': 'ok', 'service': 'flask-chart-api', 'endpoint': '/api/health'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

