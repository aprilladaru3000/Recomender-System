"""
Auto time split for Amazon interactions.
Computes percentile-based cutoffs and prints suggested --train-end/--val-end/--test-end.
Optionally writes them to a JSON for reuse.
"""
import argparse, os, json
import pandas as pd
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interactions", type=str, default="data/interactions.csv")
    ap.add_argument("--train-p", type=float, default=0.7, help="fraction for train (0-1)")
    ap.add_argument("--val-p", type=float, default=0.85, help="fraction for val end (0-1)")
    ap.add_argument("--out-json", type=str, default="data/time_split.json")
    args = ap.parse_args()

    df = pd.read_csv(args.interactions)
    ts = df["timestamp"].values
    t_train_end = int(np.quantile(ts, args.train_p))
    t_val_end   = int(np.quantile(ts, args.val_p))
    t_test_end  = int(ts.max())

    out = {"train_end": t_train_end, "val_end": t_val_end, "test_end": t_test_end}
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print("[Suggested] --train-end", t_train_end, "--val-end", t_val_end, "--test-end", t_test_end)
    print("Saved to", args.out_json)

if __name__ == "__main__":
    main()
