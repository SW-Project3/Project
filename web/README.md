# Trading System Web Dashboard

트레이딩 시스템 시각화를 위한 웹 대시보드입니다.

## 구조

```
web/
├── frontend/          # 프론트엔드 (HTML, CSS, JS)
│   ├── index.html
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── main.js
│   ├── Dockerfile
│   └── nginx.conf
├── backend/           # 백엔드 (Spring Boot)
│   ├── src/
│   │   └── main/
│   │       ├── java/com/trading/
│   │       │   ├── TradingSystemApplication.java
│   │       │   ├── controller/
│   │       │   │   └── MainController.java
│   │       │   └── config/
│   │       │       └── WebConfig.java
│   │       └── resources/
│   │           └── application.properties
│   ├── pom.xml
│   └── Dockerfile
├── docker-compose.yml
└── README.md
```

## 실행 방법

### Docker Compose 사용 (권장)

```bash
cd web
docker-compose up -d
```

- 프론트엔드: http://localhost:3000
- 백엔드 API: http://localhost:8080/api

### 개별 실행

#### 백엔드 (Spring Boot)
```bash
cd web/backend
mvn spring-boot:run
```

#### 프론트엔드 (로컬 개발)
```bash
cd web/frontend
# 간단한 HTTP 서버 실행 (Python 예시)
python -m http.server 3000
```

## 개발

### 백엔드 개발
- Java 17 이상 필요
- Maven 사용
- Spring Boot 3.2.0

### 프론트엔드 개발
- HTML, CSS, JavaScript
- Nginx로 서빙

## API 엔드포인트

- `GET /api/health` - 헬스 체크

