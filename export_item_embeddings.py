"""
export_item_embeddings.py
-------------------------
Ekspor **item embeddings** dari model Two-Tower (tower item) agar bisa di-index dengan FAISS.
Langkah:
1) Muat data & indexers.
2) Buat VisualStore (precomputed sangat disarankan).
3) Hitung embedding item: E_item = item_emb(ID) + proj(visual_feat).
4) Simpan ke NPZ: key=str(item_idx), value=vector.
"""
import os, argparse, numpy as np, pandas as pd, torch
from tqdm import tqdm
from src.runner_common import load_basic, build_indexers, make_visual_store
from src.models import TwoTower

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--encoder", type=str, choices=["cnn","vit"], default="vit")
    ap.add_argument("--precomputed", type=str, required=True)
    ap.add_argument("--out", type=str, default="data/item_emb_twotower.npz")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    inter, items, users = load_basic(args.data_root)
    inter_all, items_idxed, user2idx, item2idx = build_indexers(inter, items)

    model = TwoTower(num_users=len(user2idx), num_items=len(item2idx), id_dim=64, vis_dim=256).to(args.device)
    model.eval()
    vis_store = make_visual_store(items_idxed, encoder=args.encoder, device=args.device, precomputed_npz=args.precomputed)

    # compute item embeddings
    idxs = sorted(list(item2idx.values()))
    embs = {}
    with torch.no_grad():
        for i in tqdm(idxs, desc="Item tower emb"):
            i_t = torch.tensor([i], dtype=torch.long, device=args.device)
            v_i = vis_store(i_t)  # [1,256]
            e = model.item_tower(i_t, v_i)  # [1,64]
            embs[str(i)] = e.squeeze(0).cpu().numpy()

    np.savez_compressed(args.out, **embs)
    print("Saved item embeddings to", args.out)

if __name__ == "__main__":
    main()
