# Flask 웹 서버 실행 방법

## 설치

```bash
pip install flask flask-cors pandas numpy plotly
```

또는 프로젝트 루트의 requirements.txt에 있는 패키지들을 설치:
```bash
pip install -r requirements.txt
```

## 실행

```bash
cd web/backend
python app.py
```

서버가 `http://localhost:3000`에서 실행됩니다.

## 접속

브라우저에서 `http://localhost:3000`으로 접속하면 트레이딩 차트를 볼 수 있습니다.

## API 엔드포인트

- `GET /` - 메인 페이지 (index.html)
- `GET /api/chart-data` - 차트 데이터 JSON API

## 기능

- 9월 1일부터 10월 31일까지 BTC/USDT 15분봉 데이터 시각화
- 캔들스틱 차트
- 볼린저 밴드 오버레이
- 매수/매도 포지션 표시 (마우스 오버 시 손절가/익절가 정보 표시)
- RSI 지표
- MACD 지표

