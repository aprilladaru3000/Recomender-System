"""
Prepare Amazon dataset (reviews + metadata) into the skeleton format.
Inputs:
  --reviews <path to reviews_xxx.json.gz>  (fields: reviewerID, asin, unixReviewTime, overall, ...)
  --meta    <path to meta_xxx.json.gz>     (fields: asin, title, imUrl or imageURLHighRes, ...)
Outputs under --outdir:
  data/interactions.csv  (user_id,item_id,timestamp,label)
  data/items.csv         (item_id,image_path)
Also downloads images to data/images/ (can be large; you may limit --max-items)
"""
import os, gzip, json, argparse, pandas as pd, numpy as np, re, time, pathlib, urllib.request

def stream_json_gz(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                yield json.loads(line)
            except Exception:
                continue

def sanitize_filename(s):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', s)[:120]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviews", type=str, required=True)
    ap.add_argument("--meta", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="data")
    ap.add_argument("--min-inter", type=int, default=5, help="min interactions per user & item")
    ap.add_argument("--rating-thresh", type=float, default=4.0, help="keep interactions with rating>=thresh as positive")
    ap.add_argument("--max-items", type=int, default=5000, help="limit number of items to download images")
    ap.add_argument("--download-timeout", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    img_dir = os.path.join(args.outdir, "images")
    os.makedirs(img_dir, exist_ok=True)

    print("[1/4] Reading metadata...")
    asin2img = {}
    n_meta = 0
    for obj in stream_json_gz(args.meta):
        n_meta += 1
        asin = obj.get("asin")
        if not asin: continue
        url = None
        if isinstance(obj.get("imageURLHighRes"), list) and obj["imageURLHighRes"]:
            url = obj["imageURLHighRes"][0]
        elif obj.get("imUrl"):
            url = obj["imUrl"]
        if url:
            asin2img[asin] = url
    print(f"Loaded meta: {len(asin2img)} items with images from {n_meta} records.")

    print("[2/4] Reading reviews and filtering...")
    rows = []
    for obj in stream_json_gz(args.reviews):
        u = obj.get("reviewerID"); i = obj.get("asin")
        ts = int(obj.get("unixReviewTime", 0)); rating = float(obj.get("overall", 0))
        if not u or not i or ts<=0: continue
        if rating >= args.rating_thresh:
            rows.append((u, i, ts, 1))
    inter = pd.DataFrame(rows, columns=["user_id","item_id","timestamp","label"])
    # filter by availability of images
    inter = inter[inter["item_id"].isin(asin2img.keys())]
    # min interactions
    def apply_min(df, col, k):
        vc = df[col].value_counts()
        keep = vc[vc>=k].index
        return df[df[col].isin(keep)]
    inter = apply_min(inter, "user_id", args.min_inter)
    inter = apply_min(inter, "item_id", args.min_inter)
    inter = inter.sort_values("timestamp").reset_index(drop=True)
    print(f"Kept interactions: {len(inter)}; users={inter.user_id.nunique()} items={inter.item_id.nunique()}")

    print("[3/4] Selecting items and downloading images (may take time)...")
    uniq_items = inter["item_id"].drop_duplicates().tolist()
    if args.max_items and len(uniq_items)>args.max_items:
        uniq_items = uniq_items[:args.max_items]
        inter = inter[inter["item_id"].isin(uniq_items)].reset_index(drop=True)
    items = []
    for idx, asin in enumerate(uniq_items, 1):
        url = asin2img.get(asin)
        if not url: continue
        ext = ".jpg"
        fn = sanitize_filename(asin) + ext
        outp = os.path.join(img_dir, fn)
        if not os.path.exists(outp):
            try:
                urllib.request.urlretrieve(url, outp)
            except Exception as e:
                # skip
                continue
        if os.path.exists(outp):
            items.append((asin, os.path.relpath(outp, args.outdir)))
        if idx % 200 == 0:
            print(f"  downloaded {idx} / {len(uniq_items)}")

    items_df = pd.DataFrame(items, columns=["item_id","image_path"])
    # keep only items with downloaded images
    inter = inter[inter["item_id"].isin(set(items_df["item_id"]))].reset_index(drop=True)

    print("[4/4] Writing CSVs...")
    items_df.to_csv(os.path.join(args.outdir, "items.csv"), index=False)
    inter.to_csv(os.path.join(args.outdir, "interactions.csv"), index=False)
    # optional users.csv
    pd.DataFrame({"user_id": inter["user_id"].unique()}).to_csv(os.path.join(args.outdir, "users.csv"), index=False)
    print("Done. Files saved under", args.outdir)

if __name__ == "__main__":
    main()
