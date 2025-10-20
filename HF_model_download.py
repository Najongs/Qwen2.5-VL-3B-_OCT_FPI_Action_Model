from huggingface_hub import hf_hub_download

# 0. 설정
HF_USERNAME = "Najongs"
REPO_NAME = "Qwen2.5-VL_3B_OCT_FPI_Action_Model"
REPO_FILE_NAME = "qwen_vla_final_1000.pt" 
LOCAL_SAVE_DIR = "./checkpoints/"

FULL_REPO_ID = f"{HF_USERNAME}/{REPO_NAME}"

print(f"{FULL_REPO_ID} 저장소에서 {REPO_FILE_NAME} 파일 다운로드를 시작합니다...")

downloaded_file_path = hf_hub_download(
    repo_id=FULL_REPO_ID,
    filename=REPO_FILE_NAME,
    local_dir=LOCAL_SAVE_DIR
)

print(f"파일이 성공적으로 다운로드되었습니다. 경로: {downloaded_file_path}")