import os
import shutil
import logging
from pathlib import Path
import sys
import csv
from typing import Optional # Optional 임포트 추가

# --- 설정 ---
# 검사를 시작할 기본 경로
BASE_PATH = "/home/najo/NAS/VLA/Qwen2.5-VL-3B-_OCT_FPI_Action_Model/Real_Env_Test/"
# 허용할 시간 차이 (초)
TIME_THRESHOLD = 1.0
# 검사할 이미지 파일 확장자 (소문자)
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}

# ******** (수정) CSV에서 타임스탬프 열 인덱스 ********
# 0: recv_timestamp
# 1: origin_timestamp
# 2: send_timestamp
# "마지막 열의 인덱스 타임값" 요청에 따라 2번 인덱스(recv_timestamp)를 사용합니다.
TIMESTAMP_COLUMN_INDEX = 1
TIMESTAMP_COLUMN_NAME = "recv_timestamp" # 헤더 검사용 이름도 일치시킵니다.
# ****************************************************

# --- 로깅 설정 ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def count_image_files(dir_path: str, threshold: int = 10) -> int:
    """
    주어진 디렉토리 및 모든 하위 디렉토리를 검사하여 (os.walk 사용)
    이미지 파일의 총 개수를 셉니다.
    """
    if not os.path.isdir(dir_path):
        logging.warning(f"'{dir_path}'는 디렉토리가 아닙니다.")
        return 0

    image_count = 0
    try:
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in IMAGE_EXTENSIONS:
                    image_count += 1
                    if image_count >= threshold:
                        logging.debug(f"'{dir_path}'에서 이미지 {threshold}개 이상 발견. 카운트 중단.")
                        return image_count
                        
    except Exception as e:
        logging.error(f"'{dir_path}' 탐색 중 오류 발생: {e}")
        return 0 

    return image_count

def clean_dirs_with_few_images(base_path: str, min_images: int = 10):
    """
    base_path 바로 아래의 'recv_all_...' 폴더들을 대상으로,
    내부 이미지 파일이 'min_images' 개 미만인 경우 폴더를 삭제합니다.
    """
    if not os.path.isdir(base_path):
        logging.error(f"제공된 경로가 디렉토리가 아니거나 존재하지 않습니다: {base_path}")
        return

    logging.info(f"검사를 시작합니다: {base_path} (기준: {min_images}개 미만의 이미지 파일)")
    
    deleted_count = 0
    kept_count = 0

    try:
        for entry in os.scandir(base_path):
            if entry.is_dir() and entry.name.startswith("recv_all_"):
                dir_path = entry.path
                logging.info(f"검사 중: '{entry.name}'")
                
                image_count = count_image_files(dir_path, threshold=min_images)
                
                if image_count < min_images:
                    logging.warning(f"'{entry.name}' 폴더에 이미지가 {image_count}개만 발견되었습니다 ({min_images}개 미만). 폴더 전체를 삭제합니다...")
                    try:
                        shutil.rmtree(dir_path)
                        logging.info(f"성공적으로 삭제했습니다: {dir_path}")
                        deleted_count += 1
                    except OSError as e:
                        logging.error(f"삭제 실패 '{dir_path}': {e}")
                else:
                    logging.info(f"'{entry.name}' 폴더에 이미지가 {image_count}개 이상 존재하므로 유지합니다.")
                    kept_count += 1
            
            elif entry.is_dir():
                logging.info(f"'{entry.name}' 폴더는 'recv_all_'로 시작하지 않아 건너뜁니다.")
            elif entry.is_file():
                logging.info(f"'{entry.name}' 파일은 건너뜁니다.")
                
    except Exception as e:
        logging.error(f"디렉토리 스캔 중 예상치 못한 오류 발생: {e}")

    logging.info(f"검사 완료. 총 {deleted_count}개의 'recv_all_...' 폴더 삭제, {kept_count}개의 'recv_all_...' 폴더 유지.")

def get_start_time_from_csv(csv_path: str) -> Optional[float]:
    """
    주어진 CSV 파일에서 첫 번째 데이터 행의 타임스탬프(지정된 인덱스)를 읽어옵니다.
    """
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            try:
                header = next(reader) # 헤더 읽기
                if header[TIMESTAMP_COLUMN_INDEX].strip() != TIMESTAMP_COLUMN_NAME:
                    logging.warning(f"CSV 헤더 '{TIMESTAMP_COLUMN_NAME}'(인덱스 {TIMESTAMP_COLUMN_INDEX}) 불일치 (실제: {header[TIMESTAMP_COLUMN_INDEX]}). {csv_path}")
            except StopIteration:
                logging.warning(f"CSV 파일이 비어있습니다 (헤더 없음): {csv_path}")
                return None
            try:
                first_data_row = next(reader) # 첫 데이터 행 읽기
                timestamp_str = first_data_row[TIMESTAMP_COLUMN_INDEX]
                return float(timestamp_str)
            except StopIteration:
                logging.warning(f"CSV 파일에 데이터 행이 없습니다: {csv_path}")
                return None
            except (ValueError, IndexError) as e:
                logging.error(f"첫 데이터 행 타임스탬프(인덱스 {TIMESTAMP_COLUMN_INDEX}) 파싱 실패: {e} - {csv_path}")
                return None
    except IOError as e:
        logging.error(f"CSV 파일 읽기 오류 {csv_path}: {e}")
        return None

def extract_timestamp_from_filename(filename: str) -> Optional[float]:
    """
    'zed_..._1761209758.539.jpg' 같은 파일명에서 타임스탬프를 추출합니다.
    """
    try:
        stem = os.path.splitext(filename)[0]
        parts = stem.split('_')
        timestamp_str = parts[-1]
        return float(timestamp_str)
    except (ValueError, IndexError):
        logging.warning(f"파일명에서 타임스탬프 파싱 불가: {filename}")
        return None

def clean_images_after_timestamp(session_path: str, time_threshold: float):
    """
    세션 폴더 내의 CSV 시작 시간 + threshold 보다 늦은 이미지를 삭제합니다.
    os.walk를 사용해 모든 하위 폴더를 탐색합니다.
    """
    logging.info(f"--- 1초 이후 이미지 삭제 처리 시작: {session_path} ---")
    
    csv_files = [f for f in os.listdir(session_path) if f.startswith('robot_state_') and f.endswith('.csv')]
    if not csv_files:
        logging.warning(f"'robot_state_...' CSV 파일을 찾을 수 없습니다. 건너뜁니다.")
        return 0
    
    csv_path = os.path.join(session_path, csv_files[0])
    
    # 수정된 TIMESTAMP_COLUMN_INDEX (2)를 사용하여 시작 시간 읽기
    start_time = get_start_time_from_csv(csv_path) 
    
    if start_time is None:
        logging.error(f"CSV 시작 시간을 읽을 수 없습니다. 건너뜁니다.")
        return 0
        
    delete_threshold_time = start_time + time_threshold
    logging.info(f"CSV 기준 시간(열 {TIMESTAMP_COLUMN_INDEX}): {start_time:.3f}. 삭제 기준 시간 (1초 초과): > {delete_threshold_time:.3f}")

    deleted_count = 0
    total_files_checked = 0
    
    for root, dirs, files in os.walk(session_path):
        for filename in files:
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext not in IMAGE_EXTENSIONS:
                continue 
            
            total_files_checked += 1
            image_path = os.path.join(root, filename)
            
            image_time = extract_timestamp_from_filename(filename)
            if image_time is None:
                continue 
                
            if image_time > delete_threshold_time:
                try:
                    os.remove(image_path)
                    logging.warning(f"삭제됨 (1초 초과): {image_path} (이미지 시간: {image_time:.3f})")
                    deleted_count += 1
                except OSError as e:
                    logging.error(f"삭제 실패 {image_path}: {e}")

    logging.info(f"--- 처리 완료: 총 {total_files_checked}개 이미지 검사, {deleted_count}개 삭제됨 ---")
    return deleted_count

if __name__ == "__main__":
    
    # 삭제 기준이 되는 최소 이미지 개수 (이 값 *미만*이면 삭제)
    MIN_IMAGE_COUNT_THRESHOLD = 100 
    
    if not os.path.exists(BASE_PATH):
        logging.error(f"지정된 경로를 찾을 수 없습니다: {BASE_PATH}")
        sys.exit(1)

    # --- 작업 1: 이미지 개수 100개 미만인 폴더 삭제 ---
    logging.info(f"\n===== 1단계: 이미지 {MIN_IMAGE_COUNT_THRESHOLD}개 미만 폴더 삭제 시작 =====")
    clean_dirs_with_few_images(BASE_PATH, min_images=MIN_IMAGE_COUNT_THRESHOLD)
    logging.info("===== 1단계: 완료 =====\n")

    # --- 작업 2: 1초 이후 이미지 삭제 ---
    # (1단계에서 살아남은 폴더들을 대상으로 다시 실행됩니다)
    logging.info(f"\n===== 2단계: CSV 시작 {TIME_THRESHOLD}초 이후 이미지 삭제 시작 =====")
    
    total_deleted_all_sessions = 0
    total_sessions_processed = 0
    
    # BASE_PATH 아래의 'recv_all_...' 폴더들 순회
    try:
        for entry in os.scandir(BASE_PATH):
            if entry.is_dir() and entry.name.startswith("recv_all_"):
                total_deleted_all_sessions += clean_images_after_timestamp(entry.path, TIME_THRESHOLD)
                total_sessions_processed += 1
            elif entry.is_dir() or entry.is_file():
                # 'recv_all_'로 시작하지 않는 폴더나 파일은 로그 남기고 건너뛰기
                logging.debug(f"'{entry.name}' 항목은 2단계 처리 대상이 아니므로 건너뜁니다.")
                
    except Exception as e:
         logging.error(f"2단계 처리 중 오류 발생: {e}")

    logging.info(f"\n===== 2단계: 전체 작업 완료 =====")
    logging.info(f"총 {total_sessions_processed}개의 'recv_all_...' 폴더를 처리했습니다.")
    logging.info(f"총 {total_deleted_all_sessions}개의 1초 초과 이미지 파일을 삭제했습니다.")