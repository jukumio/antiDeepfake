from fastapi import FastAPI, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uuid
import shutil
import subprocess
from pathlib import Path

app = FastAPI()

# 경로 설정
DATA_DIR = Path("/Users/juheon/Desktop/DE_FAKE/capstone/stylegan2-ada-pytorch/InMAC/frame/data/stylegan")
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "pages"
SCRIPT_PATH = BASE_DIR / "pipe_for_api.sh"

# 폴더 없으면 생성 
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 정적 파일 mount
app.mount("/data/stylegan", StaticFiles(directory=DATA_DIR), name="stylegan-data")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("stylegan.html", {"request": request})


@app.post("/stylegan", response_class=HTMLResponse)
async def run_stylegan(request: Request, file: UploadFile):
    session_id = str(uuid.uuid4())
    session_dir = DATA_DIR / session_id
    input_dir = session_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    input_path = input_dir / file.filename
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # StyleGAN 파이프라인 실행 (비동기)
    subprocess.Popen([
        "bash", str(SCRIPT_PATH), str(input_path), str(session_dir)
    ],
        stdout=open(session_dir / "stdout.log", "w"),
        stderr=open(session_dir / "stderr.log", "w")
    )

    return templates.TemplateResponse("stylegan.html", {
        "request": request,
        "message": "StyleGAN 생성 중입니다. 수 분 뒤 결과 확인이 가능합니다.",
        "result_url": f"/result/{session_id}",
        "session_id": session_id
    })


@app.get("/result/{session_id}", response_class=HTMLResponse)
async def check_result(request: Request, session_id: str):
    session_dir = DATA_DIR / session_id
    image_path = session_dir / "fgsm_proj.png"
    video_path = session_dir / "refined" / "refine.mp4"
    log_file = session_dir / "stdout.log"
    uploaded_files = list((session_dir / "input").glob("*"))
    uploaded_filename = uploaded_files[0].name if uploaded_files else ""

    if not image_path.exists():
        log_text = ""
        if log_file.exists():
            with open(log_file, "r") as f:
                log_text = f.read()[-1000:]  # 마지막 로그 일부만

        return templates.TemplateResponse("stylegan.html", {
            "request": request,
            "message": "아직 결과가 준비되지 않았습니다. 몇 분 뒤 새로고침 해보세요.",
            "log_tail": log_text.replace("\n", "<br>")
        })

    return templates.TemplateResponse("stylegan.html", {
        "request": request,
        "result": f"/data/stylegan/{session_id}/fgsm_proj.png",
        "video": f"/data/stylegan/{session_id}/refined/refine.mp4" if video_path.exists() else None,
        "uploaded_filename": f"{session_id}/input/{uploaded_filename}"
    })


@app.get("/progress/{session_id}")
async def check_progress(session_id: str):
    session_dir = DATA_DIR / session_id
    log_file = session_dir / "stdout.log"

    if not log_file.exists():
        return {"status": "대기 중...", "log": ""}

    try:
        with open(log_file, "r") as f:
            lines = f.readlines()
        status = "진행 중..."
        if any("Generating image with FGSM..." in line for line in lines):
            status = "최종 이미지 생성 중"
        elif any("Refining selected W..." in line for line in lines):
            status = "Refinement 단계"
        elif any("Finding closest W..." in line for line in lines):
            status = "W 최적화 중"
        elif any("Projecting..." in line for line in lines):
            status = "초기 투영 중"

        tail_log = "".join(lines[-30:])  # 마지막 30줄
        return {"status": status, "log": tail_log}
    except:
        return {"status": "상태 확인 실패", "log": ""}

# 로컬에서 직접 실행 시
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
