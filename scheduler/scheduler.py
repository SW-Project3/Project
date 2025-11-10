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
        logging.FileHandler("scheduler.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

PYTHON_EXECUTABLE = sys.executable
BASE_DIR = Path(__file__).resolve().parent
DATA_PROCESS_DIR = BASE_DIR / "data_process"

LOADER_SCRIPT = DATA_PROCESS_DIR / "data_loader.py"
PROCESSOR_SCRIPT = DATA_PROCESS_DIR / "data_processor.py"

def run_loader_job():
    logging.info("--- [JOB START] data_loader.py (데이터 수집 및 저장) 시작 ---")
    try:
        result = subprocess.run(
            [PYTHON_EXECUTABLE, str(LOADER_SCRIPT)],
            capture_output=True, text=True, check=True, encoding='utf-8',
            cwd=str(DATA_PROCESS_DIR)
        )
        logging.info("data_loader.py 성공:\n" + result.stdout)

    except subprocess.CalledProcessError as e:
        logging.error(f"data_loader.py 실패!")
        logging.error("STDOUT:\n" + e.stdout)
        logging.error("STDERR:\n" + e.stderr)
    except Exception as e:
        logging.error(f"알 수 없는 오류 발생 (Loader): {e}")

    logging.info("--- [JOB END] data_loader.py 종료 ---")

def run_processor_job():
    logging.info("--- [JOB START] data_processor.py (데이터 정제) 시작 ---")
    try:
        result = subprocess.run(
            [PYTHON_EXECUTABLE, str(PROCESSOR_SCRIPT)],
            capture_output=True, text=True, check=True, encoding='utf-8',
            cwd=str(DATA_PROCESS_DIR)
        )
        logging.info("data_processor.py 성공:\n" + result.stdout)

    except subprocess.CalledProcessError as e:
        logging.error(f"data_processor.py 실패!")
        logging.error("STDOUT:\n" + e.stdout)
        logging.error("STDERR:\n" + e.stderr)
    except Exception as e:
        logging.error(f"알 수 없는 오류 발생 (Processor): {e}")

    logging.info("--- [JOB END] data_processor.py 종료 ---")

logging.info("자동화 스케줄러를 설정합니다...")

schedule.every().day.at("00:00").do(run_loader_job)
schedule.every().day.at("00:10").do(run_processor_job)

logging.info("스케줄 설정 완료. 대기 중...")
logging.info(f"  - 작업 1: 매일 00:00, {LOADER_SCRIPT.name} 실행")
logging.info(f"  - 작업 2: 매일 00:10, {PROCESSOR_SCRIPT.name} 실행")

try:
    while True:
        schedule.run_pending()
        time.sleep(1)
except KeyboardInterrupt:
    logging.info("스케줄러가 사용자에 의해 중지되었습니다.")