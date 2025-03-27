import requests

url = "http://localhost:8000/swap"
files = {
    'source': open('source.jpg', 'rb'),
    'target': open('target.jpg', 'rb')
}

response = requests.post(url, files=files)

if response.status_code == 200:
    with open("output.jpg", "wb") as f:
        f.write(response.content)
    print("결과 이미지가 output.jpg로 저장되었습니다.")
else:
    print("에러 발생:", response.json())
