import argparse, torch, numpy as np
from torch.utils.data import DataLoader
from src.runner_common import load_basic, build_indexers, make_visual_store, apply_coldstart_policy
from src.utils import timestamp_split, set_seed
from src.datasets import InteractionDataset
from src.models import NCF
from src.train import train_pairwise
from src.eval import evaluate_topk
from src.results_utils import append_result_csv, now_ts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--train-end", type=int, required=True)
    ap.add_argument("--val-end", type=int, required=True)
    ap.add_argument("--test-end", type=int, required=True)
    ap.add_argument("--encoder", type=str, choices=["cnn","vit"], default="vit")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--precomputed", type=str, default=None)
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--results-csv", type=str, default=os.path.join(args.data_root if False else 'data', 'results.csv'))
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--coldstart-policy", type=str, choices=["none","item-0shot","item-fewshot"], default="none")
    ap.add_argument("--fewshot-k", type=int, default=5)
    args = ap.parse_args()

    set_seed(42)
    inter, items, users = load_basic(args.data_root)
    inter_tr, inter_val, inter_te = timestamp_split(inter, args.train_end, args.val_end, args.test_end)
    inter_tr = apply_coldstart_policy(inter_tr, items, policy=args.coldstart-policy if False else args.coldstart_policy, fewshot_k=args.fewshot_k)

    inter_all, items_idxed, user2idx, item2idx = build_indexers(inter, items)
    merge_cols = ["user_id","item_id","timestamp"]
    inter_tr = inter_tr[merge_cols].merge(inter_all[["user_id","item_id","timestamp","user_idx","item_idx"]], on=merge_cols, how="left")
    inter_te = inter_te[merge_cols].merge(inter_all[["user_id","item_id","timestamp","user_idx","item_idx"]], on=merge_cols, how="left")

    vis_store = make_visual_store(items_idxed, encoder=args.encoder, device=args.device, precomputed_npz=args.precomputed, freeze=args.freeze_encoder)

    train_ds = InteractionDataset(inter_tr, user2idx, item2idx, items_idxed, visual_store=None, is_training=True,
                                  all_items_idx=list(item2idx.values()))
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = NCF(num_users=len(user2idx), num_items=len(item2idx), id_dim=64, vis_dim=256, hidden=256)
    model = train_pairwise(model, train_dl, vis_store, device=args.device, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay)

    user_pos = {}
    for _, r in inter_te.iterrows():
        u = int(r["user_idx"]); i = int(r["item_idx"])
        if np.isnan(u) or np.isnan(i): continue
        user_pos.setdefault(u,set()).add(i)
    all_items = list(item2idx.values())
    hr, ndcg, mrr = evaluate_topk(model, vis_store, user_pos, all_items, K=10, device=args.device)
    
print(f"HR@10={hr:.4f} NDCG@10={ndcg:.4f} MRR={mrr:.4f}")
append_result_csv(args.results_csv, {
    "ts": now_ts(),
    "model": "NCF",
    "encoder": args.encoder,
    "precomputed": str(args.precomputed),
    "train_end": args.train_end if hasattr(args, 'train_end') else 'NA',
    "val_end": args.val_end if hasattr(args, 'val_end') else 'NA',
    "test_end": args.test_end if hasattr(args, 'test_end') else 'NA',
    "coldstart_policy": args.coldstart_policy if hasattr(args, 'coldstart_policy') else 'none',
    "fewshot_k": args.fewshot_k if hasattr(args, 'fewshot_k') else 0,
    "epochs": args.epochs,
    "hr_at_10": hr,
    "ndcg_at_10": ndcg,
    "mrr": mrr
})


if __name__ == "__main__":
    main()
