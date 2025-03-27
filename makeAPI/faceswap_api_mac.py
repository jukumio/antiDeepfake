from flask import Flask, request, jsonify, send_file
import os
import subprocess
import cv2
import numpy as np
import paddle

from io import BytesIO

import sys
sys.path.append('../MobileFaceSwap')

from utils.prepare_data_o import LandmarkModel

app = Flask(__name__)

# 모델 초기화
def load_models():
    global landmarkModel
    paddle.set_device('cpu')  # Mac 환경이므로 CPU 사용
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CHECKPOINTS_DIR = os.path.join(BASE_DIR, '../MobileFaceSwap/checkpoints')
    landmarkModel = LandmarkModel(name=os.path.join(CHECKPOINTS_DIR, 'landmarks'))
    landmarkModel.prepare(ctx_id=0, det_thresh=0.6, det_size=(640,640))
    
@app.route('/swap', methods=['POST'])
def swap_faces():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    print(f"🚀 BASE_DIR 확인: {BASE_DIR}")

    CHECKPOINTS_DIR = os.path.join(BASE_DIR, '../MobileFaceSwap/checkpoints')
    print(f"🔍 CHECKPOINTS_DIR 확인: {CHECKPOINTS_DIR}")

    source_path = os.path.join(BASE_DIR, 'temp_source.jpg')
    target_path = os.path.join(BASE_DIR, 'temp_target.jpg')

    request.files['source'].save(source_path)
    request.files['target'].save(target_path)

    # 🚀 저장된 파일 확인
    if not os.path.exists(source_path) or not os.path.exists(target_path):
        return jsonify({'error': 'Uploaded images not found'}), 500
    print(f"✅ Uploaded images exist: {source_path}, {target_path}")

    # 기존 mac_image_test.py 실행
    command = [
        'python', os.path.join(BASE_DIR, '../MobileFaceSwap/mac_image_test.py'),
        '--source_img_path', os.path.abspath(source_path),
        '--target_img_path', os.path.abspath(target_path),
        '--output_dir', os.path.abspath(BASE_DIR)
    ]

    print(f"🚀 실행되는 명령어: {' '.join(command)}")

    
    try:
        result = subprocess.run(command, check=True, cwd=BASE_DIR, capture_output=True, text=True)
        print(f"✅ mac_image_test.py 실행 결과:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"❌ mac_image_test.py 실행 실패:\n{e.stderr}")
        return jsonify({'error': 'Image transformation failed', 'details': e.stderr}), 500

    result_path = os.path.join(BASE_DIR, 'temp_target.jpg')  # API가 반환할 이미지

    if not os.path.exists(result_path):
        return jsonify({'error': 'Output image not found'}), 500

    # 🚀 파일을 메모리로 읽어서 반환 (파일 깨짐 방지)
    with open(result_path, "rb") as img_file:
        img_io = BytesIO(img_file.read())

    return send_file(img_io, mimetype='image/jpeg')


if __name__ == '__main__':
    load_models()
    app.run(debug=True, host='0.0.0.0', port=8000)
