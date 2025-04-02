# save_untraced.py
import legacy
import torch
import dnnlib

pkl_path = "/Users/juheon/Desktop/DE_FAKE/capstone/stylegan/weights/ffhq.pkl"
out_path = "G_untraced.pt"

with dnnlib.util.open_url(pkl_path) as f:
    G = legacy.load_network_pkl(f)['G_ema']  # still traced
    torch.save(G, out_path)  # untraced native PyTorch model
