import os
import time
import subprocess
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "data/stylegan/input")
RESULT_DIR = os.path.join(BASE_DIR, "data/stylegan/results")
STYLE_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

app = FastAPI()
templates = Jinja2Templates(directory="pages")
app.mount("/data/stylegan/results", StaticFiles(directory=RESULT_DIR), name="static")  # result 이미지 제공
app.mount("/static", StaticFiles(directory=STYLE_DIR), name="static")  # 스타일 시트 제공
app.mount("/data/stylegan/input", StaticFiles(directory=UPLOAD_DIR), name="input_files")  # 원본 이미지 제공

@app.get("/")
async def redirect_to_stylegan():
    return RedirectResponse(url="/stylegan")

@app.get("/stylegan", response_class=HTMLResponse)
async def stylegan_form(request: Request):
    return templates.TemplateResponse("stylegan.html", {"request": request, "result": None})

@app.post("/stylegan", response_class=HTMLResponse)
async def stylegan_run(request: Request, file: UploadFile = File(...)):
    timestamp = int(time.time())
    filename = f"{timestamp}_{file.filename}"
    upload_path = os.path.join(UPLOAD_DIR, filename)

    # 파일 저장
    with open(upload_path, "wb") as f:
        f.write(await file.read())

    result_dir_for_this_run = os.path.join(RESULT_DIR, f"run_{timestamp}")
    os.makedirs(result_dir_for_this_run, exist_ok=True)

    process = subprocess.run(
        ["bash", "pipe_for_api.sh", upload_path, result_dir_for_this_run],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # 로그 저장
    log_path = os.path.join(result_dir_for_this_run, "execution.log")
    with open(log_path, "w") as log_file:
        log_file.write("=== STDOUT ===\n")
        log_file.write(process.stdout)
        log_file.write("\n\n=== STDERR ===\n")
        log_file.write(process.stderr)

    # 결과 이미지 및 비디오 경로 확인
    result_img_path = os.path.join(result_dir_for_this_run, "fgsm_proj.png")
    video_path = os.path.join(result_dir_for_this_run, "refined", "refine.mp4")

    if not os.path.exists(result_img_path):
        return HTMLResponse(content=f"<h2>결과 이미지가 생성되지 않았습니다.</h2>", status_code=500)

    # 경로 URL로 변환
    result_url = f"/data/stylegan/results/run_{timestamp}/fgsm_proj.png"
    video_url = None
    if os.path.exists(video_path):
        video_url = f"/data/stylegan/results/run_{timestamp}/refined/refine.mp4"

    # 업로드된 파일명 템플릿에 전달
    return templates.TemplateResponse("stylegan.html", {
        "request": request,
        "result": result_url,
        "video": video_url,
        "uploaded_filename": filename  # 업로드된 파일명 전달
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
