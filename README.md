Capstone Design
===============
2023.09~2025.04


사용법(mac 기준으로 작성됨)
---------------------
# Mobile Face Swap 테스트
얼굴 바꿔주는 모델이다.
실행 방법은 우선 터미널에서 파일 실행하고 터미널 하나 더 열어서 

```
curl -X POST "http://localhost:8000/swap" \
     -F "source=@source.jpg" \
     -F "target=@target.jpg" \
     -o output.jpg
```
이렇게 쳐주면 된다.

이것도 하기 귀찮으면 post_swap.py 를 써서 해보도록 하자.

# StyleGAN으로 얼굴 생성 테스트
StyleGAN2-ada-pytorch(https://github.com/NVlabs/stylegan2-ada-pytorch?tab=readme-ov-file)
기반으로 작동.

StyleGAN은 이름 그대로 기존 이미지에 수염을 달거나, 남자로 바꾼다던가 하는 
'스타일'을 바꿔주는 모델이다. 

이를 random vector로부터 점점 노이즈를 섞어가면서 새로운 이미지로 만드는 것인데,
나는 이를 이용해 기존 이미지를 보고 랜덤에서부터 생성한다면, 새 이미지가 기존과 비슷하지만
다른 점이 존재하기에 1차적으로 딥페이크를 방어할 수 있다 생각했다.

그러고 딥페이크를 뚫더라도, 사실 그건 나를 모방한 이미지를 딥페이크 한 것이기에 실질적으로는
내가 아니라는 기분을 줄 수 있다.
그리고 이것이야말로 Diffusion 같은 모델이 나오는 지금같을 떄에 가장 심리적으로 안정감을 줄 수 있는 방법이지 않을까.