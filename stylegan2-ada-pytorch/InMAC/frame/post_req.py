import requests

url = "http://localhost:8000/defake"
file_path = "/Users/juheon/Desktop/DE_FAKE/capstone/mysource/smith.jpg"

response = requests.post(
    url,
    files={"target": open(file_path, "rb")},
    data={
        "network_pkl": "/Users/juheon/Desktop/DE_FAKE/capstone/stylegan2-ada-pytorch/weights/ffhq.pkl",
        "outroot": "/Users/juheon/Desktop/DE_FAKE/capstone/results",
        "steps": "400",
        "python_script_dir": "/Users/juheon/Desktop/DE_FAKE/capstone/stylegan2-ada-pytorch/InMAC/frame",
        "work_dir": "/Users/juheon/Desktop/DE_FAKE/capstone/stylegan2-ada-pytorch"
    }
)

if response.status_code == 200:
    with open("/Users/juheon/Desktop/DE_FAKE/capstone/stylegan2-ada-pytorch/results/result.png", "wb") as f:
        f.write(response.content)
    print("이미지 저장 성공")
else:
    print("에러 발생:", response.json())
