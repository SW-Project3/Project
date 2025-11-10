"""
main/risk.py

- position_size: 포지션 크기 계산 (계좌 잔고, 위험 비율, 스탑 거리 기반)
- risk_amount_to_size: 절대 위험 금액 → 자산 단위 변환
"""

from typing import Optional


def position_size(
        balance: float,
        risk_per_trade: float,
        entry_price: float,
        stop_price: float,
        *,
        leverage: float = 1.0,
        min_position_value: float = 1.0,
        max_position_value: Optional[float] = None,
        round_to: Optional[float] = None
) -> float:
    """
    포지션 크기를 **자산 단위(예: BTC)** 기준으로 계산합니다.

    매개변수:
        - balance: 계좌 잔고 (예: USDT)
        - risk_per_trade: 거래당 위험 비율 (예: 0.01 → 잔고의 1%)
        - entry_price: 진입 가격
        - stop_price: 손절가
    옵션:
        - leverage: 레버리지 배율 (기본 1.0)
        - min_position_value: 최소 포지션 가치 (예: 1 USDT)
        - max_position_value: 최대 포지션 가치 (None이면 제한 없음)
        - round_to: 수량 반올림 단위 (예: 0.0001)

    반환값:
        - 매매할 자산의 수량 (예: BTC 0.02)
    """
    if entry_price <= 0:
        raise ValueError("entry_price must be > 0")

    stop_distance = abs(entry_price - stop_price)
    if stop_distance <= 0:
        return 0.0

    if balance <= 0 or risk_per_trade <= 0:
        return 0.0

    # 리스크 금액 = 잔고 × 거래당 위험비율
    risk_amount = balance * risk_per_trade

    # 레버리지 반영
    effective_risk_amount = risk_amount * leverage

    # 포지션 크기 = 리스크 금액 ÷ 스탑 거리
    size = effective_risk_amount / stop_distance

    # 포지션 가치 제한
    pos_value = size * entry_price
    if pos_value < min_position_value:
        return 0.0
    if max_position_value is not None and pos_value > max_position_value:
        size = max_position_value / entry_price

    # 반올림 처리
    if round_to is not None and round_to > 0:
        multiplier = 1.0 / round_to
        size = int(size * multiplier) / multiplier

    return float(max(0.0, size))


def risk_amount_to_size(
        risk_amount: float,
        entry_price: float,
        *,
        round_to: Optional[float] = None
) -> float:
    """
    절대 위험 금액(risk_amount)을 진입가(entry_price) 기준으로
    자산 수량으로 변환합니다.
    """
    if entry_price <= 0:
        raise ValueError("entry_price must be > 0")
    if risk_amount <= 0:
        return 0.0

    size = risk_amount / entry_price

    if round_to is not None and round_to > 0:
        multiplier = 1.0 / round_to
        size = int(size * multiplier) / multiplier

    return float(size)


if __name__ == "__main__":
    # 테스트 예시
    print("예시 포지션 크기:", position_size(10000, 0.01, entry_price=50000, stop_price=49500))
    print("레버리지 2배 예시:", position_size(10000, 0.01, entry_price=50000, stop_price=49500, leverage=2))
    print("risk_amount_to_size:", risk_amount_to_size(100, 50000))
