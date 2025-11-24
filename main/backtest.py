"""
main/backtest.py

- backtest_signals: 주어진 DataFrame의 'signal' 컬럼을 이용해 매우 단순한 백테스트를 수행합니다.
  (진입: signal == 1, 청산: signal == -1 또는 마지막 시점)
- 결과: 거래 내역 리스트와 최종 잔고를 반환
"""
from typing import Dict, Any, List
import pandas as pd


def backtest_signals(df: pd.DataFrame, starting_balance: float = 10000.0, slippage: float = 0.0) -> Dict[str, Any]:
    """
    매우 단순한 **롱(매수) 포지션 전용 백테스터** 입니다.
    - signal == 1 이고 현재 포지션이 없을 경우 → 전 잔고로 매수
    - signal == -1 이고 현재 포지션이 있을 경우 → 전량 매도
    - slippage: 슬리피지(가격 미끄러짐) 비율, 예: 0.001 → 0.1% 불리한 가격 반영

    반환값 (dict):
        - final_balance: 최종 잔고
        - trades: 거래 내역 리스트
        - equity_curve: 백테스트 중 잔고 변화를 나타내는 시리즈
    """
    df = df.reset_index(drop=True).copy()

    # 필요한 컬럼이 없을 경우 예외 발생
    if "close" not in df.columns or "signal" not in df.columns:
        raise ValueError("DataFrame에는 'close'와 'signal' 컬럼이 포함되어야 합니다.")

    balance = float(starting_balance)  # 초기 잔고
    position_units = 0.0              # 보유 자산 수량
    equity_curve: List[float] = []    # 시간별 자산 가치(잔고 + 보유자산)
    trades: List[Dict[str, Any]] = [] # 거래 내역 저장 리스트
    last_entry_price = None           # 마지막 진입 가격

    # 각 시점별로 시그널 확인
    for idx, row in df.iterrows():
        price = float(row["close"])
        sig = int(row.get("signal", 0))

        # 매수 신호(signal == 1)이고 현재 포지션이 없을 때 → 진입
        if sig == 1 and position_units == 0:
            buy_price = price * (1 + slippage)  # 슬리피지 반영
            position_units = balance / buy_price  # 전 잔고로 매수
            last_entry_price = buy_price
            trades.append({
                "type": "buy",
                "price": buy_price,
                "units": position_units,
                "idx": idx
            })
            balance = 0.0  # 잔고는 전부 사용

        # 매도 신호(signal == -1)이고 현재 포지션이 있을 때 → 청산
        elif sig == -1 and position_units > 0:
            sell_price = price * (1 - slippage)  # 슬리피지 반영
            balance = position_units * sell_price  # 전량 매도 후 잔고 갱신
            trades.append({
                "type": "sell",
                "price": sell_price,
                "units": position_units,
                "idx": idx,
                "pl": (sell_price - last_entry_price) * position_units if last_entry_price else None
            })
            position_units = 0.0
            last_entry_price = None

        # 현재 보유 자산 또는 잔고 기준으로 자산 가치(equity) 계산
        if position_units > 0:
            equity = position_units * price  # 포지션 보유 중 → 평가금액
        else:
            equity = balance                 # 포지션 없음 → 현금 잔고
        equity_curve.append(equity)

    # 마지막까지 포지션이 남아 있을 경우 → 마지막 가격으로 강제 청산
    if position_units > 0:
        final_price = float(df["close"].iloc[-1])
        balance = position_units * final_price
        trades.append({
            "type": "close",
            "price": final_price,
            "units": position_units,
            "idx": len(df) - 1
        })
        position_units = 0.0

    # 최종 결과 반환
    return {
        "final_balance": balance,  # 최종 잔고
        "trades": trades,          # 거래 내역
        "equity_curve": pd.Series(equity_curve, index=df.index)  # 잔고 추이
    }


