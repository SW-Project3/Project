"""
Chart Service for Web Dashboard
matplotlib를 사용하여 차트를 생성하고 base64로 인코딩하여 반환
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 백엔드에서 사용하기 위한 non-GUI 백엔드
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
from io import BytesIO
import base64
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
import json

import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
# Docker 환경에서는 /project, 로컬에서는 상대 경로 사용
if os.path.exists('/project'):
    PROJECT_ROOT = Path('/project')
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# PYTHONPATH에 추가 (맨 앞에)
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

# 디버깅: 경로 확인
print(f"[DEBUG] PROJECT_ROOT: {PROJECT_ROOT}")
print(f"[DEBUG] PROJECT_ROOT exists: {PROJECT_ROOT.exists()}")
print(f"[DEBUG] main directory exists: {(PROJECT_ROOT / 'main').exists()}")
print(f"[DEBUG] sys.path: {sys.path[:3]}")

# 이제 import 시도
try:
    # main 디렉토리 확인
    main_dir = PROJECT_ROOT / 'main'
    strategy_file = main_dir / 'strategy.py'
    indicators_file = main_dir / 'indicators.py'
    
    print(f"[DEBUG] main_dir exists: {main_dir.exists()}")
    print(f"[DEBUG] strategy.py exists: {strategy_file.exists()}")
    print(f"[DEBUG] indicators.py exists: {indicators_file.exists()}")
    
    # main 디렉토리 내용 확인
    if main_dir.exists():
        try:
            main_files = list(main_dir.iterdir())
            print(f"[DEBUG] Files in main directory: {[f.name for f in main_files[:10]]}")
        except Exception as e:
            print(f"[DEBUG] Error listing main directory: {e}")
    
    if not main_dir.exists():
        raise ImportError(f"main directory not found at {main_dir}")
    if not strategy_file.exists():
        # 실제 파일 목록 확인
        if main_dir.exists():
            actual_files = [f.name for f in main_dir.iterdir() if f.is_file()]
            raise ImportError(f"strategy.py not found at {strategy_file}. Actual files in main: {actual_files}")
        else:
            raise ImportError(f"strategy.py not found at {strategy_file}")
    
    from main.strategy import compute_indicators, generate_signals
    from main.indicators import rsi, macd, bollinger_bands, atr
    print("[DEBUG] Successfully imported main modules")
except ImportError as e:
    print(f"[ERROR] Failed to import main modules: {e}")
    print(f"[ERROR] Current working directory: {os.getcwd()}")
    print(f"[ERROR] PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"[ERROR] PROJECT_ROOT absolute: {PROJECT_ROOT.resolve()}")
    if PROJECT_ROOT.exists():
        print(f"[ERROR] Files in PROJECT_ROOT: {[f.name for f in PROJECT_ROOT.iterdir()][:10]}")
    raise


def create_candlestick_chart(
    df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    positions: Optional[pd.DataFrame] = None
) -> str:
    """
    캔들스틱 차트 생성 (볼린저 밴드 오버래핑 포함)
    포지션 및 손/익절가 표시
    
    Args:
        df: OHLCV 데이터 (Open, High, Low, Close, Volume)
        start_date: 시작 날짜 (YYYY-MM-DD)
        end_date: 종료 날짜 (YYYY-MM-DD)
        positions: 포지션 정보 (entry_price, stop_loss, take_profit 포함)
    
    Returns:
        base64 인코딩된 이미지 문자열
    """
    # 날짜 필터링
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]
    
    if len(df) == 0:
        raise ValueError("필터링된 데이터가 없습니다.")
    
    # 지표 계산
    df = compute_indicators(df)
    
    # 볼린저 밴드
    bb_ma = df['BB_MA']
    bb_upper = df['BB_Upper']
    bb_lower = df['BB_Lower']
    
    # 차트 생성
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3, 
                         left=0.08, right=0.95, top=0.95, bottom=0.08)
    
    # 메인 가격 차트 (캔들스틱 + 볼린저 밴드)
    ax_price = fig.add_subplot(gs[0:2, 0:2])
    
    # 캔들스틱 그리기
    dates = df.index
    opens = df['Open']
    highs = df['High']
    lows = df['Low']
    closes = df['Close']
    
    # 캔들 색상 결정
    colors = ['red' if closes[i] < opens[i] else 'blue' for i in range(len(df))]
    
    # 캔들 그리기
    for i in range(len(df)):
        color = colors[i]
        # 몸통
        body_bottom = min(opens.iloc[i], closes.iloc[i])
        body_top = max(opens.iloc[i], closes.iloc[i])
        body_height = body_top - body_bottom
        
        if body_height > 0:
            rect = Rectangle((i-0.3, body_bottom), 0.6, body_height, 
                           facecolor=color, edgecolor='black', alpha=0.8)
            ax_price.add_patch(rect)
        
        # 꼬리
        ax_price.plot([i, i], [lows.iloc[i], highs.iloc[i]], 
                     color='black', linewidth=0.5)
    
    # 볼린저 밴드 오버래핑
    ax_price.plot(range(len(df)), bb_upper.values, 'g--', 
                 label='BB Upper', linewidth=1, alpha=0.7)
    ax_price.plot(range(len(df)), bb_ma.values, 'g-', 
                 label='BB MA', linewidth=1, alpha=0.7)
    ax_price.plot(range(len(df)), bb_lower.values, 'g--', 
                 label='BB Lower', linewidth=1, alpha=0.7)
    ax_price.fill_between(range(len(df)), bb_upper.values, bb_lower.values, 
                         alpha=0.1, color='green')
    
    # 포지션 및 손/익절가 표시
    entry_label_added = False
    stop_loss_label_added = False
    take_profit_label_added = False
    
    if positions is not None and len(positions) > 0:
        for pos_idx, (timestamp, pos) in enumerate(positions.iterrows()):
            entry_price = pos.get('entry_price', None)
            stop_loss = pos.get('stop_loss', None)
            take_profit = pos.get('take_profit', None)
            signal = pos.get('signal', 0)
            
            # 데이터프레임에서 해당 타임스탬프의 인덱스 찾기
            try:
                time_idx = df.index.get_loc(timestamp)
            except KeyError:
                # 타임스탬프가 정확히 일치하지 않으면 가장 가까운 값 찾기
                time_idx = df.index.searchsorted(timestamp)
                if time_idx >= len(df):
                    time_idx = len(df) - 1
            
            if entry_price is not None and not pd.isna(entry_price):
                # 진입가 표시 (수평선)
                color = 'blue' if signal > 0 else 'red'
                label = 'Entry (Long)' if signal > 0 else 'Entry (Short)'
                ax_price.axhline(y=entry_price, color=color, 
                               linestyle='-', linewidth=2, alpha=0.7, 
                               label=label if not entry_label_added else '')
                entry_label_added = True
                
                # 진입 시점에 마커 표시
                ax_price.plot(time_idx, entry_price, marker='o', 
                            markersize=10, color=color, 
                            markeredgecolor='white', markeredgewidth=2,
                            zorder=5)
            
            if stop_loss is not None and not pd.isna(stop_loss):
                # 손절가 표시 (점선)
                ax_price.axhline(y=stop_loss, color='red', linestyle='--', 
                               linewidth=1.5, alpha=0.7,
                               label='Stop Loss' if not stop_loss_label_added else '')
                stop_loss_label_added = True
            
            if take_profit is not None and not pd.isna(take_profit):
                # 익절가 표시 (점선)
                ax_price.axhline(y=take_profit, color='green', linestyle='--', 
                               linewidth=1.5, alpha=0.7,
                               label='Take Profit' if not take_profit_label_added else '')
                take_profit_label_added = True
    
    ax_price.set_title('Price Chart with Bollinger Bands', fontsize=14, fontweight='bold')
    ax_price.set_ylabel('Price', fontsize=12)
    ax_price.legend(loc='upper left', fontsize=8)
    ax_price.grid(True, alpha=0.3)
    
    # RSI 차트
    ax_rsi = fig.add_subplot(gs[2, 0:2])
    rsi_values = df['RSI']
    ax_rsi.plot(range(len(df)), rsi_values.values, 'purple', linewidth=1.5)
    ax_rsi.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought')
    ax_rsi.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold')
    ax_rsi.fill_between(range(len(df)), 30, 70, alpha=0.1, color='gray')
    ax_rsi.set_title('RSI (Relative Strength Index)', fontsize=12, fontweight='bold')
    ax_rsi.set_ylabel('RSI', fontsize=10)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.legend(loc='upper right', fontsize=8)
    ax_rsi.grid(True, alpha=0.3)
    
    # MACD 차트
    ax_macd = fig.add_subplot(gs[3, 0:2])
    macd_line = df['MACD']
    macd_signal = df['MACD_Signal']
    macd_hist = df['MACD_Hist']
    
    ax_macd.plot(range(len(df)), macd_line.values, 'blue', 
                linewidth=1.5, label='MACD')
    ax_macd.plot(range(len(df)), macd_signal.values, 'red', 
                linewidth=1.5, label='Signal')
    ax_macd.bar(range(len(df)), macd_hist.values, 
               color=['green' if x > 0 else 'red' for x in macd_hist.values],
               alpha=0.6, label='Histogram')
    ax_macd.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax_macd.set_title('MACD (Moving Average Convergence Divergence)', 
                     fontsize=12, fontweight='bold')
    ax_macd.set_ylabel('MACD', fontsize=10)
    ax_macd.set_xlabel('Time', fontsize=10)
    ax_macd.legend(loc='upper right', fontsize=8)
    ax_macd.grid(True, alpha=0.3)
    
    # X축 날짜 레이블 설정 (일부만 표시)
    num_labels = min(10, len(df))
    step = len(df) // num_labels if len(df) > num_labels else 1
    x_ticks = range(0, len(df), step)
    x_labels = [df.index[i].strftime('%m/%d') for i in x_ticks]
    
    for ax in [ax_price, ax_rsi, ax_macd]:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    
    # 자산 변동 차트 (오른쪽)
    ax_equity = fig.add_subplot(gs[0:2, 2])
    # TODO: 자산 변동 데이터 추가 예정
    ax_equity.text(0.5, 0.5, 'Equity Curve\n(To be implemented)', 
                  ha='center', va='center', fontsize=12, 
                  transform=ax_equity.transAxes)
    ax_equity.set_title('Equity Curve', fontsize=12, fontweight='bold')
    ax_equity.axis('off')
    
    # 통계 정보 (오른쪽 하단)
    ax_stats = fig.add_subplot(gs[2:4, 2])
    stats_text = f"""
    Statistics
    
    Period: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}
    Data Points: {len(df)}
    
    Current Price: ${df['Close'].iloc[-1]:.2f}
    High: ${df['High'].max():.2f}
    Low: ${df['Low'].min():.2f}
    
    Current RSI: {df['RSI'].iloc[-1]:.2f}
    Current MACD: {df['MACD'].iloc[-1]:.4f}
    """
    ax_stats.text(0.1, 0.5, stats_text, ha='left', va='center', 
                 fontsize=10, family='monospace',
                 transform=ax_stats.transAxes)
    ax_stats.axis('off')
    
    # 이미지를 base64로 변환
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return img_base64


def get_chart_data(start_date: str, end_date: str) -> Dict:
    """
    차트 데이터 및 이미지 생성
    
    Args:
        start_date: 시작 날짜 (YYYY-MM-DD)
        end_date: 종료 날짜 (YYYY-MM-DD)
    
    Returns:
        차트 이미지(base64) 및 메타데이터를 포함한 딕셔너리
    """
    try:
        # 데모 CSV 데이터 로드
        # Docker 환경에서는 /project/data, 로컬에서는 상대 경로 사용
        if os.path.exists('/project/data'):
            csv_path = Path('/project/data/processed/demo_BTCUSDT_15m.csv')
        elif os.path.exists('/project'):
            # 프로젝트 루트가 마운트된 경우
            csv_path = Path('/project/data/processed/demo_BTCUSDT_15m.csv')
        else:
            CURRENT_DIR = Path(__file__).resolve().parent
            PROJECT_ROOT = CURRENT_DIR.parent.parent
            csv_path = PROJECT_ROOT / "data" / "processed" / "demo_BTCUSDT_15m.csv"
        
        if not csv_path.exists():
            raise FileNotFoundError(f"데모 데이터 파일을 찾을 수 없습니다: {csv_path}")
        
        # CSV 파일 로드
        df = pd.read_csv(csv_path)
        
        # 컬럼명 소문자로 정규화
        df.columns = [col.lower() for col in df.columns]
        
        # 타임스탬프를 인덱스로 설정
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)
        df.sort_values(by='timestamp', inplace=True)
        df.set_index('timestamp', inplace=True)
        
        # 인덱스를 timezone-naive로 변환 (비교 오류 방지)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        # 컬럼명 대문자로 변환 (기존 코드와 호환)
        df.columns = [col.capitalize() for col in df.columns]
        
        # 날짜 필터링 (모두 naive datetime으로 통일)
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        df = df[(df.index >= start_dt) & (df.index <= end_dt)]
        
        if len(df) == 0:
            raise ValueError(f"{start_date} ~ {end_date} 기간의 데이터가 없습니다.")
        
        # 포지션 정보 가져오기 (신호 생성)
        # Prophet 예측값이 없으므로 더미 예측값 생성 (실제로는 model에서 가져와야 함)
        try:
            # 더미 Prophet 예측값 생성
            prophet_forecast = pd.DataFrame({
                'ds': df.index,
                'yhat': df['Close'] * (1 + np.random.randn(len(df)) * 0.01),
                'yhat_lower': df['Close'] * 0.995,
                'yhat_upper': df['Close'] * 1.005
            })
            
            # 신호 생성
            signals = generate_signals(df, prophet_forecast)
            
            # 포지션이 있는 신호만 필터링 (인덱스는 이미 df.index와 일치)
            positions = signals[signals['signal'] != 0].copy()
            if len(positions) == 0:
                positions = None
        except Exception as e:
            print(f"Warning: Could not generate positions: {e}")
            positions = None
        
        # 차트 생성
        chart_image = create_candlestick_chart(df, start_date, end_date, positions)
        
        return {
            'success': True,
            'chart_image': chart_image,
            'start_date': start_date,
            'end_date': end_date,
            'data_points': len(df),
            'current_price': float(df['Close'].iloc[-1]),
            'high': float(df['High'].max()),
            'low': float(df['Low'].min()),
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

