
'''
실행 방법
uvicorn InMAC.frame.defake_api:app --reload --host 0.0.0.0 --port 8000
'''

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
import os
import subprocess
import uuid

app = FastAPI()

@app.post("/defake")
async def defake_image(
    target: UploadFile = File(...),
    network_pkl: str = Form(...),
    outroot: str = Form(...),
    steps: int = Form(default=400),
    python_script_dir: str = Form(...), #상대경로
    work_dir: str = Form(...)
):
    session_id = str(uuid.uuid4())[:8]

    # 작업 디렉토리 설정
    session_outroot = os.path.join(outroot, session_id)
    w_candidates_dir = os.path.join(session_outroot, "w_candidates")
    os.makedirs(w_candidates_dir, exist_ok=True)

    # 업로드된 타겟 이미지 저장
    target_path = os.path.join(work_dir, f"target_{session_id}.jpg")
    with open(target_path, "wb") as f:
        f.write(await target.read())

    if not os.path.exists(network_pkl):
        return JSONResponse(status_code=500, content={"error": "network_pkl not found"})

    # PYTHONPATH 설정
    env = os.environ.copy()
    env["PYTHONPATH"] = work_dir

    try:
        # Step 1: 여러 W 생성
        seeds = subprocess.check_output(["shuf", "-i", "0-10000", "-n", "3"]).decode().split()
        for seed in seeds:
            subprocess.run([
                "python", f"{python_script_dir}/projector_mps_W.py",
                "--network", network_pkl,
                "--target", target_path,
                "--outdir", f"{w_candidates_dir}/seed{seed}",
                "--seed", seed,
                "--save-video", "false",
                "--num-steps", str(steps),
                "--use-mps"
            ], check=True, env=env)

        # Step 2: 가장 가까운 W 찾기
        closest_w_path = os.path.join(session_outroot, "closest_w.npz")
        subprocess.run([
            "python", f"{python_script_dir}/find_closest_w.py",
            "--target", target_path,
            "--w_candidates", w_candidates_dir,
            "--network", network_pkl,
            "--outpath", closest_w_path,
            "--use_mps"
        ], check=True, env=env)

        # Step 2.5: W refinement
        refined_dir = os.path.join(session_outroot, "refined")
        subprocess.run([
            "python", f"{python_script_dir}/refine.py",
            "--network", network_pkl,
            "--target", target_path,
            "--w-init", closest_w_path,
            "--outdir", refined_dir,
            "--num-steps", "200",
            "--initial-lr", "0.008",
            "--betas", "0.85", "0.98",
            "--lpips-weight", "0.6",
            "--reg-noise-weight", "20000",
            "--noise-mode", "random",
            "--use-mps",
            "--save-video"
        ], check=True, env=env)

        # Step 3: FGSM 공격 후 이미지 생성
        subprocess.run([
            "python", f"{python_script_dir}/generate_fgsm.py",
            "--network", network_pkl,
            "--w", f"{refined_dir}/refined_w.npz",
            "--target", target_path,
            "--outdir", session_outroot,
            "--epsilon", "0.05",
            "--use-mps"
        ], check=True, env=env)

    except subprocess.CalledProcessError as e:
        return JSONResponse(status_code=500, content={"error": e.stderr})

    # 결과 이미지 반환
    fgsm_path = os.path.join(session_outroot, "fgsm_proj.png")
    if not os.path.exists(fgsm_path):
        return JSONResponse(status_code=500, content={"error": "FGSM output not found"})

    return JSONResponse(content={"fgsm_path": fgsm_path})
    # path를 리턴해줌