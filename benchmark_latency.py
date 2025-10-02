"""
benchmark_latency.py
--------------------
Ukur **latensi inferensi** per 1.000 item untuk model rekomendasi berbasis ID+visual.
Fokus pada *item scoring* agar bisa diperbandingkan antara CNN vs ViT (precomputed).

Langkah yang dilakukan skrip ini:
1) Memuat data & mapping indeks (user_idx, item_idx).
2) Memuat model yang dipilih (VBPR/NCF/Two-Tower) dalam mode evaluasi.
3) Membangun callable visual store (pakai --precomputed agar fair & cepat).
4) Mengukur waktu untuk menghitung skor 1 user terhadap N item (dengan batching),
   dan melaporkan latensi per 1.000 item (ms), throughput item/detik, dan memori model (MB).
"""
import os, argparse, time, psutil, torch, numpy as np, pandas as pd
from torch.utils.data import DataLoader
from src.runner_common import load_basic, build_indexers, make_visual_store
from src.models import VBPR, NCF, TwoTower

def model_memory_mb(model: torch.nn.Module) -> float:
    """Kira-kira memory parameter + buffer (MB)."""
    tot = 0
    for p in model.parameters():
        tot += p.numel() * p.element_size()
    for b in model.buffers():
        tot += b.numel() * b.element_size()
    return tot / (1024*1024)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--model", type=str, choices=["vbpr","ncf","two_tower"], default="two_tower")
    ap.add_argument("--encoder", type=str, choices=["cnn","vit"], default="vit", help="hanya dipakai untuk label/visual store")
    ap.add_argument("--precomputed", type=str, required=True, help="NPZ fitur visual precomputed")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--repeat", type=int, default=5, help="ulangi pengukuran untuk rata-rata")
    args = ap.parse_args()

    inter, items, users = load_basic(args.data_root)
    inter_all, items_idxed, user2idx, item2idx = build_indexers(inter, items)

    num_users, num_items = len(user2idx), len(item2idx)
    # siapkan model
    if args.model == "vbpr":
        model = VBPR(num_users, num_items, id_dim=64, vis_dim=256)
    elif args.model == "ncf":
        model = NCF(num_users, num_items, id_dim=64, vis_dim=256, hidden=256)
    else:
        model = TwoTower(num_users, num_items, id_dim=64, vis_dim=256)
    model = model.to(args.device)
    model.eval()

    # visual store precomputed (WAJIB untuk benchmark fair & cepat)
    vis_store = make_visual_store(items_idxed, encoder=args.encoder, device=args.device, precomputed_npz=args.precomputed)

    # ambil 1 user acak yang punya interaksi (untuk tower user)
    sample_u = int(inter_all["user_idx"].dropna().sample(1, random_state=42).values[0])
    all_items = np.array(list(item2idx.values()), dtype=np.int64)
    u_tensor_all = torch.full((len(all_items),), sample_u, dtype=torch.long, device=args.device)
    i_tensor_all = torch.tensor(all_items, dtype=torch.long, device=args.device)

    # warmup
    with torch.no_grad():
        s = 0
        for k in range(0, len(all_items), args.batch_size):
            batch = {
                "user_idx": u_tensor_all[k:k+args.batch_size],
                "pos_item_idx": i_tensor_all[k:k+args.batch_size],
                "neg_item_idx": i_tensor_all[k:k+args.batch_size],
            }
            model(batch, vis_store)

    # ukur
    times = []
    with torch.no_grad():
        for _ in range(args.repeat):
            t0 = time.time()
            for k in range(0, len(all_items), args.batch_size):
                batch = {
                    "user_idx": u_tensor_all[k:k+args.batch_size],
                    "pos_item_idx": i_tensor_all[k:k+args.batch_size],
                    "neg_item_idx": i_tensor_all[k:k+args.batch_size],
                }
                model(batch, vis_store)
            t1 = time.time()
            times.append(t1 - t0)

    avg = np.mean(times)
    per_1k_ms = (avg / len(all_items)) * 1000 * 1000  # ms per 1000 items
    throughput = len(all_items) / avg  # items per second

    print(f"Model: {args.model} | Encoder(feat): {args.encoder} | Items: {len(all_items)}")
    print(f"Latency: {per_1k_ms:.2f} ms per 1k items | Throughput: {throughput:.1f} items/s | Repeat: {args.repeat}")
    print(f"Model params+buffers ~= {model_memory_mb(model):.1f} MB")

if __name__ == "__main__":
    main()
