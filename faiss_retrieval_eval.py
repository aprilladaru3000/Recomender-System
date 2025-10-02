"""
faiss_retrieval_eval.py
-----------------------
Evaluasi Top-N dengan **retrieval ANN (FAISS)** untuk Two-Tower:
1) Muat Two-Tower (untuk user tower).
2) Muat item embeddings (NPZ) dan FAISS index (opsional: kalau tidak ada, fallback brute-force numpy).
3) Untuk setiap user di test, ambil kandidat Top-K dari ANN, hitung HR@K / NDCG@K / MRR.

Catatan: Ini menggantikan brute-force ranking pada evaluate_topk agar scalable.
"""
import os, argparse, numpy as np, pandas as pd, torch, faiss
from tqdm import tqdm
from src.runner_common import load_basic, build_indexers, make_visual_store
from src.utils import timestamp_split, set_seed
from src.models import TwoTower

def load_keys(mapping_path):
    keys = []
    with open(mapping_path, "r") as f:
        for line in f:
            k = line.strip()
            if k:
                keys.append(int(k))
    return np.array(keys, dtype=np.int64)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--train-end", type=int, required=True)
    ap.add_argument("--val-end", type=int, required=True)
    ap.add_argument("--test-end", type=int, required=True)
    ap.add_argument("--encoder", type=str, choices=["cnn","vit"], default="vit")
    ap.add_argument("--precomputed", type=str, required=True)
    ap.add_argument("--item-emb", type=str, required=True, help="NPZ keyed by item_idx (str) -> vector")
    ap.add_argument("--faiss-index", type=str, required=True)
    ap.add_argument("--faiss-keys", type=str, required=True)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    set_seed(42)
    inter, items, users = load_basic(args.data_root)
    inter_tr, inter_val, inter_te = timestamp_split(inter, args.train_end, args.val_end, args.test_end)

    inter_all, items_idxed, user2idx, item2idx = build_indexers(inter, items)
    # remap test
    merge_cols = ["user_id","item_id","timestamp"]
    inter_te = inter_te[merge_cols].merge(inter_all[["user_id","item_id","timestamp","user_idx","item_idx"]], on=merge_cols, how="left")

    # build user->pos in test
    user_pos = {}
    for _, r in inter_te.iterrows():
        if np.isnan(r["user_idx"]) or np.isnan(r["item_idx"]): continue
        u = int(r["user_idx"]); i = int(r["item_idx"])
        user_pos.setdefault(u,set()).add(i)

    # load model (only user_tower is needed)
    model = TwoTower(num_users=len(user2idx), num_items=len(item2idx), id_dim=64, vis_dim=256).to(args.device)
    model.eval()
    vis_store = make_visual_store(items_idxed, encoder=args.encoder, device=args.device, precomputed_npz=args.precomputed)

    # load item embeddings and faiss index
    arrs = np.load(args.item_emb)
    keys = load_keys(args.faiss_keys)        # np.array of item_idx (ints) in same row order as index
    index = faiss.read_index(args.faiss_index)

    # metrics
    hits, ndcgs, mrrs = [], [], []
    with torch.no_grad():
        for u, pos_items in tqdm(user_pos.items(), desc="FAISS Eval"):
            # compute user vector
            u_t = torch.tensor([u], dtype=torch.long, device=args.device)
            pu = model.user_tower(u_t)  # [1,64]
            q = pu.cpu().numpy().astype("float32")  # [1,64]
            # search ANN
            D, I = index.search(q, args.K)  # I: indices into keys array
            cand_items = keys[I[0]].tolist()

            # HR@K
            hit = 1.0 if any(i in pos_items for i in cand_items) else 0.0
            hits.append(hit)
            # NDCG@K (single rel)
            dcg = 0.0
            for rank, idx in enumerate(cand_items, start=1):
                rel = 1.0 if idx in pos_items else 0.0
                dcg += rel / np.log2(rank + 1)
            idcg = 1.0 if len(pos_items) > 0 else 0.0
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
            # MRR
            rr = 0.0
            for rank, idx in enumerate(cand_items, start=1):
                if idx in pos_items:
                    rr = 1.0 / rank; break
            mrrs.append(rr)

    print(f"HR@{args.K}={np.mean(hits):.4f} NDCG@{args.K}={np.mean(ndcgs):.4f} MRR={np.mean(mrrs):.4f}")

if __name__ == "__main__":
    main()
