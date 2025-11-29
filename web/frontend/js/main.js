// Main JavaScript file
// 도커 환경에서는 nginx를 통해 프록시되므로 상대 경로 사용
const API_BASE_URL = '/api';

// 날짜 범위 (전체 기간)
const FULL_START_DATE = '2025-09-01';
const FULL_END_DATE = '2025-10-31';

// 현재 선택된 날짜 범위
let currentStartDate = FULL_START_DATE;
let currentEndDate = FULL_END_DATE;

document.addEventListener('DOMContentLoaded', function() {
    console.log('Trading System Dashboard loaded');
    
    // 약간의 지연을 두고 날짜 필터 설정 (DOM이 완전히 로드되도록)
    setTimeout(function() {
        setupDateFilter();
        // Initialize application
        init();
    }, 100);
});

function setupDateFilter() {
    const startDateInput = document.getElementById('start-date');
    const endDateInput = document.getElementById('end-date');
    const applyBtn = document.getElementById('apply-filter');
    const resetBtn = document.getElementById('reset-filter');
    
    if (!startDateInput || !endDateInput || !applyBtn || !resetBtn) {
        console.error('날짜 필터 요소를 찾을 수 없습니다.');
        return;
    }
    
    console.log('날짜 필터 설정 완료');
    
    // 시작 날짜가 종료 날짜보다 늦으면 종료 날짜를 조정
    startDateInput.addEventListener('change', function() {
        if (this.value > endDateInput.value) {
            endDateInput.value = this.value;
        }
    });
    
    // 종료 날짜가 시작 날짜보다 이전이면 시작 날짜를 조정
    endDateInput.addEventListener('change', function() {
        if (this.value < startDateInput.value) {
            startDateInput.value = this.value;
        }
    });
    
    // 적용 버튼
    applyBtn.addEventListener('click', function(e) {
        e.preventDefault();
        console.log('적용 버튼 클릭됨');
        
        const newStartDate = startDateInput.value;
        const newEndDate = endDateInput.value;
        
        console.log('선택된 날짜:', newStartDate, '~', newEndDate);
        
        if (!newStartDate || !newEndDate) {
            alert('시작 날짜와 종료 날짜를 모두 선택해주세요.');
            return;
        }
        
        if (newStartDate > newEndDate) {
            alert('시작 날짜가 종료 날짜보다 늦을 수 없습니다.');
            return;
        }
        
        console.log('날짜 필터 적용:', newStartDate, '~', newEndDate);
        currentStartDate = newStartDate;
        currentEndDate = newEndDate;
        loadChartData();
    });
    
    // 전체 기간 버튼
    resetBtn.addEventListener('click', function(e) {
        e.preventDefault();
        console.log('전체 기간 버튼 클릭됨');
        
        startDateInput.value = FULL_START_DATE;
        endDateInput.value = FULL_END_DATE;
        currentStartDate = FULL_START_DATE;
        currentEndDate = FULL_END_DATE;
        loadChartData();
    });
}

async function init() {
    try {
        await loadChartData();
    } catch (error) {
        console.error('초기화 오류:', error);
        showError('차트 데이터를 불러오는 중 오류가 발생했습니다: ' + error.message);
    }
}

async function loadChartData() {
    const loadingEl = document.getElementById('loading');
    const chartContainer = document.getElementById('chart-container');
    const errorEl = document.getElementById('error');
    
    try {
        loadingEl.style.display = 'block';
        chartContainer.style.display = 'none';
        errorEl.style.display = 'none';
        
        // 날짜 범위 파라미터 추가
        const url = `${API_BASE_URL}/chart-data?start_date=${currentStartDate}&end_date=${currentEndDate}`;
        console.log('API 호출 시작:', url);
        
        const response = await fetch(url);
        
        console.log('응답 상태:', response.status, response.statusText);
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error('응답 오류:', errorText);
            throw new Error(`HTTP ${response.status}: ${errorText}`);
        }
        
        const data = await response.json();
        console.log('데이터 수신 완료:', data);
        
        if (!data.success) {
            console.error('API 오류:', data.error);
            throw new Error(data.error || '데이터 로드 실패');
        }
        
        loadingEl.style.display = 'none';
        chartContainer.style.display = 'block';
        
        createChart(data);
        
    } catch (error) {
        console.error('데이터 로드 오류:', error);
        console.error('에러 스택:', error.stack);
        loadingEl.style.display = 'none';
        showError('차트 데이터를 불러오는 중 오류가 발생했습니다: ' + error.message);
    }
}

function createChart(data) {
    const priceData = data.price;
    const bbData = data.bollinger_bands;
    const rsiData = data.rsi;
    const macdData = data.macd;
    const buyPositions = data.buy_positions || [];
    const sellPositions = data.sell_positions || [];
    
    // 가격 범위 계산 (Y축 범위를 80%로 설정)
    const prices = priceData.close.filter(p => p > 0);
    if (prices.length === 0) {
        showError('가격 데이터가 없습니다.');
        return;
    }
    
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    const priceRange = maxPrice - minPrice;
    
    // Y축 범위를 80%로 설정 (상하 각각 10% 여백)
    // 최소값은 데이터의 최소값부터 시작 (0이 아닌 실제 최소값)
    const padding = priceRange * 0.1;
    let yAxisMin = minPrice - padding;
    let yAxisMax = maxPrice + padding;
    
    // 최소값이 너무 낮아지지 않도록 보정 (데이터 최소값의 95% 이상)
    const minAllowed = minPrice * 0.95;
    if (yAxisMin < minAllowed) {
        yAxisMin = minAllowed;
    }
    
    console.log('가격 범위:', minPrice.toFixed(2), '~', maxPrice.toFixed(2));
    console.log('Y축 범위:', yAxisMin.toFixed(2), '~', yAxisMax.toFixed(2));
    
    // 날짜 범위에 따른 제목 설정
    const titleDateRange = currentStartDate === FULL_START_DATE && currentEndDate === FULL_END_DATE
        ? '2025-09-01 ~ 2025-10-31'
        : `${currentStartDate} ~ ${currentEndDate}`;
    
    // 서브플롯 생성
    const traces = [];
    const layout = {
        title: `BTC/USDT 트레이딩 차트 (${titleDateRange})`,
        height: 1000,
        xaxis: {
            domain: [0, 1],
            rangeslider: { visible: false }
        },
        xaxis2: {
            domain: [0, 1],
            anchor: 'y2'
        },
        xaxis3: {
            domain: [0, 1],
            anchor: 'y3'
        },
        yaxis: {
            domain: [0.6, 1],
            title: '가격 (USDT)',
            range: [yAxisMin, yAxisMax],  // Y축 범위 설정
            autorange: false,  // 자동 범위 비활성화
            fixedrange: false  // 고정 범위 비활성화 (줌 가능)
        },
        yaxis2: {
            domain: [0.3, 0.58],
            title: 'RSI',
            range: [0, 100]
        },
        yaxis3: {
            domain: [0, 0.28],
            title: 'MACD'
        },
        hovermode: 'x unified',
        template: 'plotly_white',
        showlegend: true
    };
    
    // 1. 캔들스틱 차트
    traces.push({
        x: priceData.timestamp,
        open: priceData.open,
        high: priceData.high,
        low: priceData.low,
        close: priceData.close,
        type: 'candlestick',
        name: '가격',
        increasing: { line: { color: '#26a69a' } },
        decreasing: { line: { color: '#ef5350' } },
        xaxis: 'x',
        yaxis: 'y'
    });
    
    // 2. 볼린저 밴드
    traces.push({
        x: bbData.timestamp,
        y: bbData.upper,
        type: 'scatter',
        mode: 'lines',
        name: 'BB 상단',
        line: { color: 'rgba(128,128,128,0.5)', width: 1, dash: 'dash' },
        xaxis: 'x',
        yaxis: 'y',
        showlegend: true
    });
    
    traces.push({
        x: bbData.timestamp,
        y: bbData.middle,
        type: 'scatter',
        mode: 'lines',
        name: 'BB 중심선',
        line: { color: 'rgba(128,128,128,0.7)', width: 1 },
        xaxis: 'x',
        yaxis: 'y',
        showlegend: true
    });
    
    traces.push({
        x: bbData.timestamp,
        y: bbData.lower,
        type: 'scatter',
        mode: 'lines',
        name: 'BB 하단',
        line: { color: 'rgba(128,128,128,0.5)', width: 1, dash: 'dash' },
        fill: 'tonexty',
        fillcolor: 'rgba(128,128,128,0.1)',
        xaxis: 'x',
        yaxis: 'y',
        showlegend: true
    });
    
    // 3. 매수 포지션
    if (buyPositions.length > 0) {
        const buyTimes = buyPositions.map(p => p.timestamp);
        const buyPrices = buyPositions.map(p => p.price);
        const buyHoverTexts = buyPositions.map(p => 
            `<b>매수</b><br>` +
            `시간: ${new Date(p.timestamp).toLocaleString('ko-KR')}<br>` +
            `진입가: ${p.entry_price.toFixed(2)} USDT<br>` +
            `손절가: ${p.stop_loss.toFixed(2)} USDT<br>` +
            `익절가: ${p.take_profit.toFixed(2)} USDT<br>` +
            `손절 거리: ${((p.entry_price - p.stop_loss) / p.entry_price * 100).toFixed(2)}%<br>` +
            `익절 거리: ${((p.take_profit - p.entry_price) / p.entry_price * 100).toFixed(2)}%`
        );
        
        traces.push({
            x: buyTimes,
            y: buyPrices,
            type: 'scatter',
            mode: 'markers',
            name: '매수',
            marker: {
                symbol: 'triangle-up',
                size: 12,
                color: '#26a69a',
                line: { width: 2, color: 'white' }
            },
            text: buyHoverTexts,
            hovertemplate: '%{text}<extra></extra>',
            xaxis: 'x',
            yaxis: 'y',
            showlegend: true,
            customdata: buyPositions.map(p => ({
                stop_loss: p.stop_loss,
                take_profit: p.take_profit,
                entry_price: p.entry_price,
                timestamp: p.timestamp
            }))
        });
    }
    
    // 4. 매도 포지션
    if (sellPositions.length > 0) {
        const sellTimes = sellPositions.map(p => p.timestamp);
        const sellPrices = sellPositions.map(p => p.price);
        const sellHoverTexts = sellPositions.map(p => 
            `<b>매도</b><br>` +
            `시간: ${new Date(p.timestamp).toLocaleString('ko-KR')}<br>` +
            `진입가: ${p.entry_price.toFixed(2)} USDT<br>` +
            `손절가: ${p.stop_loss.toFixed(2)} USDT<br>` +
            `익절가: ${p.take_profit.toFixed(2)} USDT<br>` +
            `손절 거리: ${((p.stop_loss - p.entry_price) / p.entry_price * 100).toFixed(2)}%<br>` +
            `익절 거리: ${((p.entry_price - p.take_profit) / p.entry_price * 100).toFixed(2)}%`
        );
        
        traces.push({
            x: sellTimes,
            y: sellPrices,
            type: 'scatter',
            mode: 'markers',
            name: '매도',
            marker: {
                symbol: 'triangle-down',
                size: 12,
                color: '#ef5350',
                line: { width: 2, color: 'white' }
            },
            text: sellHoverTexts,
            hovertemplate: '%{text}<extra></extra>',
            xaxis: 'x',
            yaxis: 'y',
            showlegend: true,
            customdata: sellPositions.map(p => ({
                stop_loss: p.stop_loss,
                take_profit: p.take_profit,
                entry_price: p.entry_price,
                timestamp: p.timestamp
            }))
        });
    }
    
    // 5. RSI 지표
    traces.push({
        x: rsiData.timestamp,
        y: rsiData.rsi,
        type: 'scatter',
        mode: 'lines',
        name: 'RSI',
        line: { color: 'purple', width: 2 },
        xaxis: 'x2',
        yaxis: 'y2',
        showlegend: false
    });
    
    // RSI 과매수/과매도 라인
    layout.shapes = layout.shapes || [];
    layout.shapes.push({
        type: 'line',
        x0: rsiData.timestamp[0],
        x1: rsiData.timestamp[rsiData.timestamp.length - 1],
        y0: 70,
        y1: 70,
        xref: 'x2',
        yref: 'y2',
        line: { color: 'red', width: 1, dash: 'dash' },
        opacity: 0.5
    });
    
    layout.shapes.push({
        type: 'line',
        x0: rsiData.timestamp[0],
        x1: rsiData.timestamp[rsiData.timestamp.length - 1],
        y0: 30,
        y1: 30,
        xref: 'x2',
        yref: 'y2',
        line: { color: 'green', width: 1, dash: 'dash' },
        opacity: 0.5
    });
    
    // 6. MACD 지표
    traces.push({
        x: macdData.timestamp,
        y: macdData.macd,
        type: 'scatter',
        mode: 'lines',
        name: 'MACD',
        line: { color: 'blue', width: 2 },
        xaxis: 'x3',
        yaxis: 'y3',
        showlegend: false
    });
    
    traces.push({
        x: macdData.timestamp,
        y: macdData.signal,
        type: 'scatter',
        mode: 'lines',
        name: 'Signal',
        line: { color: 'orange', width: 2 },
        xaxis: 'x3',
        yaxis: 'y3',
        showlegend: false
    });
    
    // MACD 히스토그램
    const histColors = macdData.hist.map(val => val >= 0 ? 'green' : 'red');
    traces.push({
        x: macdData.timestamp,
        y: macdData.hist,
        type: 'bar',
        name: 'MACD Hist',
        marker: { color: histColors },
        xaxis: 'x3',
        yaxis: 'y3',
        showlegend: false
    });
    
    // 0 라인
    layout.shapes.push({
        type: 'line',
        x0: macdData.timestamp[0],
        x1: macdData.timestamp[macdData.timestamp.length - 1],
        y0: 0,
        y1: 0,
        xref: 'x3',
        yref: 'y3',
        line: { color: 'gray', width: 1, dash: 'dash' },
        opacity: 0.5
    });
    
    // 기존 차트 제거 후 새로 그리기
    const chartDiv = document.getElementById('trading-chart');
    Plotly.purge(chartDiv);
    
    // 차트 그리기
    Plotly.newPlot('trading-chart', traces, layout, {
        responsive: true,
        displayModeBar: true
    }).then(function() {
        // 마우스 오버 이벤트 핸들러 추가
        setupHoverLines('trading-chart', buyPositions, sellPositions);
    });
    
    // 자산 변동 차트 그리기
    if (data.asset_curve) {
        createAssetChart(data.asset_curve);
    }
    
    console.log('차트 생성 완료. Y축 범위:', yAxisMin, '~', yAxisMax);
}

function setupHoverLines(chartId, buyPositions, sellPositions) {
    const chartDiv = document.getElementById(chartId);
    let currentShapes = [];
    
    chartDiv.on('plotly_hover', function(data) {
        // 기존 라인 제거
        if (currentShapes.length > 0) {
            const update = {
                shapes: currentShapes.map(() => null)
            };
            Plotly.relayout(chartId, update);
            currentShapes = [];
        }
        
        // 호버된 포인트 확인
        if (data.points && data.points.length > 0) {
            const point = data.points[0];
            const pointIndex = point.pointNumber;
            
            // 매수 포지션 확인
            if (point.data.name === '매수' && point.data.customdata && point.data.customdata[pointIndex]) {
                const pos = point.data.customdata[pointIndex];
                const shapes = [
                    {
                        type: 'line',
                        x0: pos.timestamp,
                        x1: pos.timestamp,
                        y0: pos.stop_loss,
                        y1: pos.stop_loss,
                        xref: 'x',
                        yref: 'y',
                        line: { color: 'red', width: 2, dash: 'dot' },
                        opacity: 0.8
                    },
                    {
                        type: 'line',
                        x0: pos.timestamp,
                        x1: pos.timestamp,
                        y0: pos.take_profit,
                        y1: pos.take_profit,
                        xref: 'x',
                        yref: 'y',
                        line: { color: 'green', width: 2, dash: 'dot' },
                        opacity: 0.8
                    }
                ];
                Plotly.relayout(chartId, { shapes: shapes });
                currentShapes = shapes;
            }
            
            // 매도 포지션 확인
            if (point.data.name === '매도' && point.data.customdata && point.data.customdata[pointIndex]) {
                const pos = point.data.customdata[pointIndex];
                const shapes = [
                    {
                        type: 'line',
                        x0: pos.timestamp,
                        x1: pos.timestamp,
                        y0: pos.stop_loss,
                        y1: pos.stop_loss,
                        xref: 'x',
                        yref: 'y',
                        line: { color: 'red', width: 2, dash: 'dot' },
                        opacity: 0.8
                    },
                    {
                        type: 'line',
                        x0: pos.timestamp,
                        x1: pos.timestamp,
                        y0: pos.take_profit,
                        y1: pos.take_profit,
                        xref: 'x',
                        yref: 'y',
                        line: { color: 'green', width: 2, dash: 'dot' },
                        opacity: 0.8
                    }
                ];
                Plotly.relayout(chartId, { shapes: shapes });
                currentShapes = shapes;
            }
        }
    });
    
    chartDiv.on('plotly_unhover', function() {
        // 호버 해제 시 라인 제거
        if (currentShapes.length > 0) {
            const update = {
                shapes: currentShapes.map(() => null)
            };
            Plotly.relayout(chartId, update);
            currentShapes = [];
        }
    });
}

function createAssetChart(assetData) {
    if (!assetData || !assetData.timestamp || !assetData.asset_value) {
        console.warn('자산 데이터가 없습니다.');
        return;
    }
    
    // 자산 가치 범위 계산
    const assetValues = assetData.asset_value.filter(v => v > 0);
    if (assetValues.length === 0) {
        console.warn('자산 가치 데이터가 없습니다.');
        return;
    }
    
    const minAsset = Math.min(...assetValues);
    const maxAsset = Math.max(...assetValues);
    const assetRange = maxAsset - minAsset;
    const padding = assetRange * 0.1;
    const yAxisMin = Math.max(0, minAsset - padding);
    const yAxisMax = maxAsset + padding;
    
    const trace = {
        x: assetData.timestamp,
        y: assetData.asset_value,
        type: 'scatter',
        mode: 'lines',
        name: '자산 가치',
        line: {
            color: '#26a69a',
            width: 2
        },
        fill: 'tozeroy',
        fillcolor: 'rgba(38, 166, 154, 0.1)'
    };
    
    const returnColor = assetData.return_pct >= 0 ? '#26a69a' : '#ef5350';
    
    const layout = {
        title: {
            text: `자산 변동<br><span style="font-size: 11px; color: ${returnColor};">초기: $${assetData.initial_cash.toFixed(2)} → 최종: $${assetData.final_value.toFixed(2)}<br>수익률: ${assetData.return_pct >= 0 ? '+' : ''}${assetData.return_pct.toFixed(2)}%</span>`,
            font: { size: 14 }
        },
        height: 1000,
        xaxis: {
            title: '시간',
            showgrid: true,
            gridcolor: 'rgba(128,128,128,0.2)'
        },
        yaxis: {
            title: '자산 가치 (USD)',
            range: [yAxisMin, yAxisMax],
            autorange: false,
            showgrid: true,
            gridcolor: 'rgba(128,128,128,0.2)'
        },
        hovermode: 'x unified',
        template: 'plotly_white',
        showlegend: false,
        margin: { l: 60, r: 30, t: 80, b: 60 }
    };
    
    // 초기 자산 라인 추가
    layout.shapes = [{
        type: 'line',
        x0: assetData.timestamp[0],
        x1: assetData.timestamp[assetData.timestamp.length - 1],
        y0: assetData.initial_cash,
        y1: assetData.initial_cash,
        xref: 'paper',
        yref: 'y',
        line: { color: 'gray', width: 1, dash: 'dash' },
        opacity: 0.5
    }];
    
    const chartDiv = document.getElementById('asset-chart');
    if (chartDiv) {
        Plotly.purge(chartDiv);
        Plotly.newPlot('asset-chart', [trace], layout, {
            responsive: true,
            displayModeBar: true
        });
    }
}

function showError(message) {
    const errorEl = document.getElementById('error');
    errorEl.textContent = message;
    errorEl.style.display = 'block';
}
