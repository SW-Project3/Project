"""
Flask API for Chart Service
"""
import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python path에 추가 (chart_service import 전에)
if os.path.exists('/project'):
    PROJECT_ROOT = Path('/project')
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from flask import Flask, jsonify, request
from flask_cors import CORS
from chart_service import get_chart_data

app = Flask(__name__)
CORS(app)  # CORS 허용


@app.route('/api/chart/data', methods=['GET'])
def chart_data():
    """차트 데이터 및 이미지 반환"""
    start_date = request.args.get('startDate', '2025-09-01')
    end_date = request.args.get('endDate', '2025-10-31')
    
    print(f"[API] Chart data request: {start_date} ~ {end_date}")
    
    try:
        result = get_chart_data(start_date, end_date)
        print(f"[API] Chart data generated: success={result.get('success')}, has_image={bool(result.get('chart_image'))}")
        if result.get('chart_image'):
            print(f"[API] Chart image length: {len(result['chart_image'])}")
        return jsonify(result)
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[API] Error generating chart: {e}")
        print(f"[API] Traceback: {error_trace}")
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': error_trace
        }), 500


@app.route('/api/health', methods=['GET'])
def health():
    """헬스 체크"""
    return jsonify({
        'status': 'ok',
        'service': 'chart-api'
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

