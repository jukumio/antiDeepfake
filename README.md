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
stylengan2-ada-pytorch 폴더 들어가서
```
uvicorn InMAC.frame.defake_api:app --reload --host 0.0.0.0 --port 8000
```

실행하면 fastapi로 변환해둔 실행기가 켜진다. 중요한 점은 ffhq.pkl이라는 모델을 저장해둔 파일이 없으면 안되는데,

```
python projector.py --outdir=out --target=~/mytargetimg.png \
--network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
```
이걸 실행하거나 링크 들어가서 다운받도록 하자.

그 다음으로 터미널을 하나 더 열어서 post_req 파일을 실행하면 된다. 여기서 파라미터에 본인 위치를 잘 입력하도록 하자.


굳이 api로 안해도 된다 그러면 pipeline_runner라는 bash 파일을 열어서 여기다 파라미터 넣고 하면 된다. 개인적으로 이게 더 편하다.

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
pipeline 속 projector는 랜덤 5개 시드로부터 가중치 w를 구하고, 그 중 가장 근접한 w값을 구해
새롭게 generate 시키면서 동시에 fgsm 필터를 씌우는 것이다.