"""
Precompute visual embeddings for items into a NPZ file.
Usage:
  python scripts/precompute_visual.py --data-root data --encoder vit --out data/vit_emb.npz --freeze-encoder
"""
import os, argparse, numpy as np, pandas as pd, torch
from tqdm import tqdm
from src.visual_store import VisualIndexer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--encoder", type=str, choices=["cnn","vit"], default="vit")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    items = pd.read_csv(os.path.join(args.data_root, "items.csv"))
    enc = VisualIndexer(encoder=args.encoder, out_dim=256, freeze=args.freeze_encoder).to(args.device)
    cache = {}
    for _, r in tqdm(items.iterrows(), total=len(items), desc="Precompute"):
        path = r["image_path"]
        if not isinstance(path, str) or not os.path.isfile(os.path.join(args.data_root, path)):
            continue
        abs_path = os.path.join(args.data_root, path)
        with torch.no_grad():
            v = enc.encode_image(abs_path).cpu().numpy()
        cache[str(r["item_id"])] = v
    # save as item_idx later after indexing; for simplicity we save keyed by item_id here
    np.savez_compressed(args.out, **cache)
    print("Saved embeddings to", args.out)

if __name__ == "__main__":
    main()
