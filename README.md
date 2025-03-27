Capstone Design
===============
2023.09~2025.04


사용법(mac 기준으로 작성됨)
---------------------
# Mobile Face Swap 테스트
얼굴 바꿔주는 모델이다.
실행 방법은 우선 터미널에서 파일 실행하고 터미널 하나 더 열어서 

₩₩₩
curl -X POST "http://localhost:8000/swap" \
     -F "source=@source.jpg" \
     -F "target=@target.jpg" \
     -o output.jpg
₩₩₩
이렇게 쳐주면 된다.

이것도 하기 귀찮으면 post_swap.py 를 써서 해보도록 하자.
