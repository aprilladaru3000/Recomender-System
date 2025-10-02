"""
build_faiss_index.py
--------------------
Bangun index FAISS dari item embeddings Two-Tower dan simpan ke file.
"""
import os, argparse, numpy as np, faiss

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--item-emb", type=str, required=True, help="NPZ keyed by item_idx (str) -> vector")
    ap.add_argument("--out-index", type=str, default="data/faiss.index")
    args = ap.parse_args()

    arrs = np.load(args.item_emb)
    keys = sorted(arrs.files, key=lambda x: int(x))
    X = np.stack([arrs[k] for k in keys], axis=0).astype("float32")
    d = X.shape[1]
    index = faiss.index_factory(d, "IVF4096,Flat")  # simple IVF
    index.nprobe = 16
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, 4096, faiss.METRIC_L2)
    index.train(X)
    index.add(X)
    faiss.write_index(index, args.out_index)
    # Save mapping of row->item_idx
    mapping_path = args.out_index + ".keys.txt"
    with open(mapping_path, "w") as f:
        for k in keys:
            f.write(k + "\n")
    print("Saved FAISS index to", args.out_index, "and keys to", mapping_path)

if __name__ == "__main__":
    main()
