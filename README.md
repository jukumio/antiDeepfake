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
먼저 envs에 있는 macstyle.yml 파일을 conda로 설치한다. 
추가적으로 fast api 관련 설정도 설치해야 웹 사이트 접속이 가능하다.

환경 구성이 끝났다면 stylengan2-ada-pytorch/InMAC/frame 폴더 들어가서
```
python stylegan_app.py
```
를 실행하면 로컬 호스트로 접속이 가능해진다. 그리고 원하는 이미지를 넣으면 파이프라인이 작동한다.


난 시연 시에 ngrox를 통해 일시적으로 로컬 호스트를 웹으로 띄웠다.

아니면 pipeline_runner라는 bash 파일을 열어서 파라미터를 알맞게 설정해 실행할 수도 있다. 
개인적으로 이게 더 편하다.

```
bash pipeline_runner.sh
```

# StyleGAN 변조기 설명
StyleGAN2-ada-pytorch(https://github.com/NVlabs/stylegan2-ada-pytorch?tab=readme-ov-file)
기반으로 작동.

StyleGAN은 이름 그대로 기존 이미지에 수염을 달거나, 남자로 바꾼다던가 하는 
'스타일'을 바꿔주는 모델이다. 

이 모델은 어떤 스타일끼리는 서로 관계가 맺어져 있다고 보고 있다. 
예를 들면, 여자와 긴 머리, 남자와 수염 같은 관계 말이다.

거기서 아이디어를 따와 이 모델은 처음엔 시드값을 받아서 latent space를 구축하지만 그 다음부터는
1차로 돌렸던 모델의 분포를 사용해 intermediate latent space를 사용하게 된다.

난 여기서 아이디어를 얻어 1차적으로 latent space를 구축하고, 그걸 바탕으로 필터를 씌운 새로운
이미지로 재탄생시키기로 했다.
pipeline 속 projector는 랜덤 3개 시드로부터 Latent space W를 구하고, 그 중 LPIPS 거리가 
가장 짧은 이미지를 생성한 projector의 W를 가져와 새롭게 이미지를 생성함과 동시에 FGSM 필터를 씌운다.