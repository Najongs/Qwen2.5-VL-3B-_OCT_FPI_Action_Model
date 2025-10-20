from huggingface_hub import HfApi

# 0. 설정
HF_USERNAME = "Najongs"
REPO_NAME = "Qwen2.5-VL_3B_OCT_FPI_Action_Model"
LOCAL_FILE_PATH = "checkpoints/qwen_vla_final_1000.pt" # 실제 파일 경로
REPO_FILE_NAME = "qwen_vla_final_1000.pt" # 저장소에 올라갈 파일 이름

# 1. API 객체 생성
api = HfApi()

# 2. 파일 업로드
print(f"Uploading {LOCAL_FILE_PATH} to {REPO_NAME}...")
api.upload_file(
    path_or_fileobj=LOCAL_FILE_PATH,    # 로컬에 있는 파일 경로
    path_in_repo=REPO_FILE_NAME,        # 허브 저장소에 저장될 경로/이름
    repo_id=f"{HF_USERNAME}/{REPO_NAME}", # "아이디/저장소이름"
    repo_type="model"                   # 모델 저장소라고 명시
)
print("Upload complete!")


# save_safetensors=True