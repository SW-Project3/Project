# scheduler.py (cwd 제거 최종본)

import schedule
import time
import subprocess
import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scheduler.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

# --- 경로 설정 (네 스크린샷 기준) ---
PYTHON_EXECUTABLE = sys.executable
# 현재 파일(scheduler.py)의 폴더 (Project/scheduler)
CURRENT_DIR = Path(__file__).resolve().parent
# 프로젝트 루트 폴더 (Project/)
PROJECT_ROOT = CURRENT_DIR.parent
# data_process 폴더 경로 (Project/data_process)
DATA_PROCESS_DIR = PROJECT_ROOT / "data_process"

# 실행할 스크립트들의 정확한 경로
LOADER_SCRIPT = DATA_PROCESS_DIR / "data_loader.py"
PROCESSOR_SCRIPT = DATA_PROCESS_DIR / "data_processor.py"
# --- --- --- --- --- --- --- ---

def run_loader_job():
    logging.info("--- [JOB START] data_loader.py (데이터 수집 및 저장) 시작 ---")
    try:
        # ★★★★★ cwd 인수를 제거 ★★★★★
        result = subprocess.run(
            [PYTHON_EXECUTABLE, str(LOADER_SCRIPT)],
            capture_output=True, text=True, check=True, encoding='utf-8'
        )
        logging.info("data_loader.py 성공:\n" + result.stdout)
        return True

    except subprocess.CalledProcessError as e:
        logging.error(f"data_loader.py 실패!")
        logging.error("STDOUT:\n" + e.stdout)
        logging.error("STDERR:\n" + e.stderr)
        return False
    except Exception as e:
        logging.error(f"알 수 없는 오류 발생 (Loader): {e}")
        return False

    logging.info("--- [JOB END] data_loader.py 종료 ---")

def run_processor_job():
    logging.info("--- [JOB START] data_processor.py (데이터 정제) 시작 ---")
    try:
        # ★★★★★ cwd 인수를 제거 ★★★★★
        result = subprocess.run(
            [PYTHON_EXECUTABLE, str(PROCESSOR_SCRIPT)],
            capture_output=True, text=True, check=True, encoding='utf-8'
        )
        logging.info("data_processor.py 성공:\n" + result.stdout)

    except subprocess.CalledProcessError as e:
        logging.error(f"data_processor.py 실패!")
        logging.error("STDOUT:\n" + e.stdout)
        logging.error("STDERR:\n" + e.stderr)
    except Exception as e:
        logging.error(f"알 수 없는 오류 발생 (Processor): {e}")

    logging.info("--- [JOB END] data_processor.py 종료 ---")

def run_pipeline_job():
    logging.info("=== [PIPELINE START] 15분 주기 작업 시작 ===")
    loader_success = run_loader_job()
    if loader_success:
        logging.info("Loader 작업 성공. 10초 후 Processor 작업 시작...")
        time.sleep(10)
        run_processor_job()
    else:
        logging.error("Loader 작업 실패. Processor 작업을 실행하지 않습니다.")
    logging.info("=== [PIPELINE END] 15분 주기 작업 종료 ===")

logging.info("자동화 스케줄러를 설정합니다...")
schedule.every(15).minutes.do(run_pipeline_job)
logging.info("스케줄 설정 완료. 대기 중...")
logging.info(f"  - 작업: 매 15분마다 run_pipeline_job (loader -> processor) 실행")

run_pipeline_job()

try:
    while True:
        schedule.run_pending()
        time.sleep(1)
except KeyboardInterrupt:
    logging.info("스케줄러가 사용자에 의해 중지되었습니다.")