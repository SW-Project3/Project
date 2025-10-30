"""
main/trade.py

- PaperTrader: 페이퍼 트레이딩(모의 투자) 시뮬레이터 (주문 기록용)
- LiveTrader: ccxt 기반 최소 래퍼 클래스 (실거래 시 반드시 검증 필요)
"""

from typing import Dict, Any, Optional
import logging
from dataclasses import dataclass, field
import time

# 로깅 설정
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)
if not LOG.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    LOG.addHandler(ch)


@dataclass
class PaperOrder:
    """
    페이퍼 트레이딩용 주문 정보를 담는 데이터 클래스
    """
    symbol: str        # 거래 심볼 (예: "BTC/USDT")
    side: str          # 매매 방향 ('buy' 또는 'sell')
    amount: float      # 주문 수량
    price: Optional[float] = None   # 주문 가격 (지정가 주문의 경우)
    timestamp: float = field(default_factory=time.time)  # 주문 시간
    meta: Dict[str, Any] = field(default_factory=dict)   # 기타 메타데이터


class PaperTrader:
    """
    단순한 페이퍼 트레이딩(모의 거래) 시뮬레이터 클래스.
    실제 거래소 연결 없이 주문 내역을 기록만 함.
    """

    def __init__(self):
        self.orders: list[PaperOrder] = []  # 주문 내역 저장 리스트

    def place_order(self, symbol: str, side: str, amount: float, price: Optional[float] = None) -> PaperOrder:
        """
        새로운 주문 생성 및 기록
        """
        order = PaperOrder(symbol=symbol, side=side, amount=amount, price=price)
        self.orders.append(order)
        LOG.info(f"[PaperTrader] 주문 생성: {order}")
        return order

    def get_orders(self):
        """
        저장된 모든 주문 내역 반환
        """
        return list(self.orders)


# LiveTrader: 실제 거래소(ccxt)를 이용한 최소 기능 래퍼
class LiveTrader:
    def __init__(self, exchange):
        """
        exchange: ccxt 거래소 인스턴스 (동기식)
        - 사용자가 직접 생성 및 인증을 완료해야 함.

        예시:
            import ccxt
            exchange = ccxt.binance({'apiKey': KEY, 'secret': SECRET})
        """
        self.exchange = exchange

    def place_market_order(self, symbol: str, side: str, amount: float) -> Dict[str, Any]:
        """
        시장가 주문 실행
        ⚠️ 주의: 실제 거래를 실행하므로 반드시 테스트(Sandbox) 후에만 사용해야 함.
        """
        side = side.lower()
        if side not in ("buy", "sell"):
            raise ValueError("side 값은 'buy' 또는 'sell' 이어야 합니다.")
        LOG.info(f"[LiveTrader] 시장가 주문 실행: {symbol} {side} {amount}")
        if side == "buy":
            return self.exchange.create_market_buy_order(symbol, amount)
        else:
            return self.exchange.create_market_sell_order(symbol, amount)

    def place_limit_order(self, symbol: str, side: str, amount: float, price: float) -> Dict[str, Any]:
        """
        지정가 주문 실행
        """
        side = side.lower()
        LOG.info(f"[LiveTrader] 지정가 주문 실행: {symbol} {side} {amount} @ {price}")
        if side == "buy":
            return self.exchange.create_limit_buy_order(symbol, amount, price)
        else:
            return self.exchange.create_limit_sell_order(symbol, amount, price)


if __name__ == "__main__":
    # 페이퍼 트레이딩 예시 실행
    pt = PaperTrader()
    pt.place_order("BTC/USDT", "buy", 0.001, price=30000)
    pt.place_order("BTC/USDT", "sell", 0.001, price=31000)
    print("주문 내역:", pt.get_orders())
