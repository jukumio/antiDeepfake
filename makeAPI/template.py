from flask import Flask, request, jsonify
import os
import subprocess

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_SCRIPT = os.path.join(BASE_DIR, 'main.py')

@app.route('/run_main', methods=['POST'])
def run_main():
    try:
        config_path = os.path.join(BASE_DIR, 'config.json')
        
        with open(config_path, 'w') as f:
            f.write(request.data.decode('utf-8'))
        
        command = ['python', MAIN_SCRIPT]
        
        result = subprocess.run(command, check=True, cwd=BASE_DIR, capture_output=True, text=True)
        
        return jsonify({
            'message': 'main.py 실행 완료',
            'stdout': result.stdout,
            'stderr': result.stderr
        })
    except subprocess.CalledProcessError as e:
        return jsonify({'error': 'main.py 실행 실패', 'details': e.stderr}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
