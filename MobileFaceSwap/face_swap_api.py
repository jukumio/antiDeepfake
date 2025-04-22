from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import shutil, os, uuid
import cv2
import numpy as np
import paddle
import sys

# 경로 설정
sys.path.append('./MobileFaceSwap')  # 경로는 실제 경로에 맞게 수정

from utils.prepare_data_o import LandmarkModel
from utils.align_face import align_img, dealign
from models.model import FaceSwap, l2_norm
from models.arcface import IRBlock, ResNet
from utils.util import cv2paddle, paddle2cv

# 앱 초기화
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/output", StaticFiles(directory="output"), name="output")
app.mount("/uploaded", StaticFiles(directory="uploaded"), name="uploaded")


# 경로 설정 및 디렉토리 생성
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'checkpoints')
os.makedirs("uploaded", exist_ok=True)
os.makedirs("output", exist_ok=True)

# 모델 로딩 (앱 시작 시 1회만)
landmarkModel = LandmarkModel(name=os.path.join(CHECKPOINTS_DIR, 'landmarks'))
landmarkModel.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640))

faceswap_model = FaceSwap(False)
id_net = ResNet(block=IRBlock, layers=[3, 4, 23, 3])
id_net.set_dict(paddle.load(os.path.join(CHECKPOINTS_DIR, 'arcface.pdparams')))
id_net.eval()
weight = paddle.load(os.path.join(CHECKPOINTS_DIR, 'MobileFaceSwap_224.pdparams'))
faceswap_model.eval()


# 얼굴 정렬 함수
def face_align(image_path, image_size=224):
    img = cv2.imread(image_path)
    landmark = landmarkModel.get(img)
    if landmark is not None:
        base_path = image_path.rsplit('.', 1)[0]
        aligned_img, back_matrix_data = align_img(img, landmark, image_size)
        aligned_path = base_path + "_aligned.png"
        back_path = base_path + "_back.npy"
        cv2.imwrite(aligned_path, aligned_img)
        np.save(back_path, back_matrix_data)
        return aligned_path, back_path
    else:
        raise ValueError(f"얼굴 감지 실패: {image_path}")


# ID 임베딩 추출 함수
def get_id_emb(id_net, aligned_img_path):
    id_img = cv2.imread(aligned_img_path)
    id_img = cv2.resize(id_img, (112, 112))
    id_img = cv2paddle(id_img)
    mean = paddle.to_tensor([[0.485, 0.456, 0.406]]).reshape((1, 3, 1, 1))
    std = paddle.to_tensor([[0.229, 0.224, 0.225]]).reshape((1, 3, 1, 1))
    id_img = (id_img - mean) / std
    id_emb, id_feature = id_net(id_img)
    return l2_norm(id_emb), id_feature


# 홈 페이지
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result_image": None})


# 업로드 및 얼굴 스왑 처리
@app.post("/upload/", response_class=HTMLResponse)
async def upload(request: Request, source_img: UploadFile = File(...), target_img: UploadFile = File(...)):
    uid = uuid.uuid4().hex
    src_path = f"uploaded/source_{uid}.jpg"
    tgt_path = f"uploaded/target_{uid}.jpg"
    out_path = f"output/result_{uid}.jpg"

    # 저장
    with open(src_path, "wb") as f:
        shutil.copyfileobj(source_img.file, f)
    with open(tgt_path, "wb") as f:
        shutil.copyfileobj(target_img.file, f)

    try:
        # 얼굴 정렬
        src_aligned, _ = face_align(src_path)
        tgt_aligned, tgt_back = face_align(tgt_path)

        # 임베딩 추출 및 설정
        id_emb, id_feature = get_id_emb(id_net, src_aligned)
        faceswap_model.set_model_param(id_emb, id_feature, model_weight=weight)

        # 타겟 이미지 로드 및 변환
        origin_att_img = cv2.imread(tgt_path)
        att_img = cv2.imread(tgt_aligned)
        att_img = cv2paddle(att_img)

        res, mask = faceswap_model(att_img)
        res = paddle2cv(res)

        if res is None or res.shape[0] == 0:
            raise ValueError("변환된 이미지가 비어 있음!")

        # 결과 복원
        back_matrix_data = np.load(tgt_back)
        mask = np.transpose(mask[0].numpy(), (1, 2, 0))
        res = dealign(res, origin_att_img, back_matrix_data, mask)

        cv2.imwrite(out_path, res)

        return templates.TemplateResponse("index.html", {
            "request": request,
            "source_image": f"/{src_path}",
            "target_image": f"/{tgt_path}",
            "result_image": f"/output/result_{uid}.jpg"
        })


    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "result_image": None,
            "error": f"에러 발생: {str(e)}"
        })
