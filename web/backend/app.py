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

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from main import indicators
from main.indicator_strategy_connector import generate_signals_indicator_only, DEFAULT_PARAMS

app = Flask(__name__, static_folder='../frontend', static_url_path='')
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


def load_data():
    """데이터 로드 및 캐싱"""
    global cached_data, cached_signals
    
    if cached_data is not None:
        return cached_data, cached_signals
    
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


@app.route('/')
def index():
    """메인 페이지"""
    return send_from_directory('../frontend', 'index.html')


@app.route('/api/chart-data')
def get_chart_data():
    """차트 데이터 API"""
    try:
        df, signals = load_data()
        
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
        
        return jsonify({
            'success': True,
            'price': price_data,
            'bollinger_bands': bb_data,
            'rsi': rsi_data,
            'macd': macd_data,
            'buy_positions': buy_positions,
            'sell_positions': sell_positions
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

