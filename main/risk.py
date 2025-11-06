"""
main/risk.py

- position_size: 단순한 포지션 사이즈 계산 (계좌 잔고, 위험 비율, 스탑 거리 기반)
- risk_amount_to_size: 보조 함수 (레버리지나 계약 단위 고려 가능)
"""
from typing import Union


def position_size(balance: float, risk_per_trade: float, entry_price: float, stop_price: float) -> float:
    """
    포지션 크기를 **자산 단위(예: BTC)** 기준으로 계산합니다.
    각 거래에서 감수할 위험 금액은 `balance * risk_per_trade` 입니다.

    매개변수:
        - balance: 계좌 잔고 (예: USDT)
        - risk_per_trade: 거래당 위험 비율 (예: 0.01 → 잔고의 1%)
        - entry_price: 진입 가격 (단위 자산당 가격)
        - stop_price: 손절가 (단위 자산당 가격)

    반환값:
        - 매매할 자산의 수량(단위)
          (예: BTC 0.02개처럼)
          만약 스탑 거리(stop_distance)가 0 이하이면 0 반환
    """
    if entry_price <= 0:
        raise ValueError("entry_price must be > 0")  # 진입가가 0 이하이면 예외 발생

    # 진입가와 손절가의 차이 (리스크 거리)
    stop_distance = abs(entry_price - stop_price)

    # 스탑 거리가 0 이하일 경우 포지션 크기를 0으로 설정
    if stop_distance <= 0:
        return 0.0

    # 계좌 잔고에서 이번 거래에 감수할 금액 (리스크 금액)
    risk_amount = balance * float(risk_per_trade)

    # 리스크 금액을 스탑 거리로 나누어 포지션 크기 계산
    size = risk_amount / stop_distance

    return float(size)


def risk_amount_to_size(risk_amount: float, entry_price: float) -> float:
    """
    절대 위험 금액(risk_amount)을 진입가(entry_price) 기준으로
    자산 수량(단위)으로 변환합니다.

    매개변수:
        - risk_amount: 위험 금액 (예: USDT 단위)
        - entry_price: 진입 가격 (예: BTC의 현재가)

    반환값:
        - 포지션 크기 (자산 단위)
    """
    if entry_price <= 0:
        raise ValueError("entry_price must be > 0")  # 진입가가 0 이하이면 예외 발생

    return risk_amount / entry_price


if __name__ == "__main__":
    # 예시 실행
    print("예시 포지션 크기:", position_size(10000, 0.01, entry_price=50000, stop_price=49500))
