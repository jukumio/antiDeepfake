#!/bin/bash

# ----------- 사용자 설정 -----------
NETWORK_PKL="/Users/juheon/Desktop/DE_FAKE/capstone/stylegan2-ada-pytorch/weights/ffhq.pkl"
TARGET_IMG="/Users/juheon/Desktop/DE_FAKE/capstone/mysource/smith.jpg"
OUTROOT="/Users/juheon/Desktop/DE_FAKE/capstone/results"
STEPS=400
PYTHON_SCRIPT_DIR="/Users/juheon/Desktop/DE_FAKE/capstone/stylegan2-ada-pytorch/InMAC/frame"
export PYTHONPATH="/Users/juheon/Desktop/DE_FAKE/capstone/stylegan2-ada-pytorch"
mkdir -p "$OUTROOT/w_candidates"

# ----------- 1단계: 여러 w 생성 -----------
# 랜덤하게 5개 seed 선택
SEED_LIST=($(shuf -i 0-10000 -n 5))  

for SEED in "${SEED_LIST[@]}"; do
    echo "[Seed ${SEED}] Projecting..."
    python projector_mps_W.py \
        --network "$NETWORK_PKL" \
        --target "$TARGET_IMG" \
        --outdir "$OUTROOT/w_candidates/seed${SEED}" \
        --seed $SEED \
        --save-video false \
        --num-steps $STEPS \
        --use-mps
done

# ----------- 2단계: 가장 가까운 w 찾기 -----------
echo "[Step 2] Finding closest W..."
python "${PYTHON_SCRIPT_DIR}/find_closest_w.py" \
    --target "$TARGET_IMG" \
    --w_candidates "$OUTROOT/w_candidates" \
    --network "$NETWORK_PKL" \
    --outpath "$OUTROOT/closest_w.npz" \
    --use_mps

# ----------- 2.5단계: 선택된 w 다시 projector로 refinement -----------
echo "[Step 2.5] Refining selected W..."
python "${PYTHON_SCRIPT_DIR}/refine.py" \
    --network "$NETWORK_PKL" \
    --target "$TARGET_IMG" \
    --w-init "$OUTROOT/closest_w.npz" \
    --outdir "$OUTROOT/refined" \
    --num-steps 200 \
    --initial-lr 0.008 \
    --betas 0.85 0.98 \
    --lpips-weight 0.6 \
    --reg-noise-weight 20000 \
    --noise-mode random \
    --use-mps \
    --save-video

# ----------- 3단계: FGSM 공격 후 생성 -----------
echo "[Step 3] Generating image with FGSM..."
python "${PYTHON_SCRIPT_DIR}/generate_fgsm.py" \
    --network "$NETWORK_PKL" \
    --w "$OUTROOT/refined/refined_w.npz" \
    --target "$TARGET_IMG" \
    --outdir "$OUTROOT" \
    --epsilon 0.05 \
    --use-mps
# --------- 4단계: 결과 저장 -----------
echo "Output saved in $OUTROOT"
