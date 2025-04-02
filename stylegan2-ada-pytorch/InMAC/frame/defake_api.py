from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
import os
import subprocess
import shutil
import uuid

app = FastAPI()

@app.post("/defake")
async def defake_image(target: UploadFile = File(...)):
    session_id = str(uuid.uuid4())[:8]
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 작업 디렉토리
    work_dir = os.path.join(base_dir, "workspace", session_id)
    os.makedirs(work_dir, exist_ok=True)

    target_path = os.path.join(work_dir, "target.jpg")
    with open(target_path, "wb") as f:
        f.write(await target.read())

    # 네트워크 파일
    network_pkl = os.path.join(base_dir, "weights", "ffhq.pkl")
    if not os.path.exists(network_pkl):
        return JSONResponse(status_code=500, content={"error": "ffhq.pkl not found"})

    # 쉘 스크립트 경로
    script_path = os.path.join(base_dir, "run_defake_pipeline.sh")

    # 실행
    try:
        subprocess.run(
            ["bash", script_path, network_pkl, target_path, work_dir],
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        return JSONResponse(status_code=500, content={"error": e.stderr})

    # 최종 이미지 경로
    fgsm_path = os.path.join(work_dir, "fgsm_proj.png")
    if not os.path.exists(fgsm_path):
        return JSONResponse(status_code=500, content={"error": "fgsm output not found"})

    return FileResponse(fgsm_path, media_type="image/png")

