import requests
import os


# 경로는 본인이 직접 수정해야 함.
url = "http://localhost:8000/defake"
file_path = "/Users/juheon/Desktop/DE_FAKE/capstone/mysource/smith.jpg"

response = requests.post(
    url,
    files={"target": open(file_path, "rb")},
    data={
        "network_pkl": "/Users/juheon/Desktop/DE_FAKE/capstone/stylegan2-ada-pytorch/weights/ffhq.pkl",
        "outroot": "/Users/juheon/Desktop/DE_FAKE/capstone/results",
        "steps": "400",
        "work_dir": "/Users/juheon/Desktop/DE_FAKE/capstone/stylegan2-ada-pytorch",
        "python_script_dir": "InMAC/frame"
    }
)


if response.status_code == 200:
    fgsm_path = response.json()["fgsm_path"]
    print("FGSM 이미지 경로:", fgsm_path)
    print("이미지 저장 성공!")  # 또는 다운로드 코드 추가
else:
    print("에러 발생:", response.json())
