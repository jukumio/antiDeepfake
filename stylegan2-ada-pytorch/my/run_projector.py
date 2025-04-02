import sys
import os
import subprocess

# Conda 환경의 python 경로
python_path = "/opt/anaconda3/envs/stylemac/bin/python"

# 현재 디렉토리를 stylegan 기준으로 세팅
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# projector_mac_2_refine.py 경로
script_path = os.path.join(project_root, 'my', 'projector_mac_2_refine.py')

# 필요한 인자들 세팅
network_pkl = os.path.join(project_root, 'weights', 'ffhq_nojit.pkl')
target_image = os.path.join(project_root, '..', 'mysource', 'smith.jpg')
output_dir = os.path.join(project_root, '..', 'results')

# 실행 명령어 구성
cmd = [
    python_path, 
    script_path,
    "--network", network_pkl,
    "--target", target_image,
    "--outdir", output_dir,
    "--save-video", "True"
]

# 어떤 python 쓰는지 출력
print("실행에 사용할 Python:", python_path)
subprocess.run(cmd)
