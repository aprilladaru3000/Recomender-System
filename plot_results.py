"""
Plot curves from data/results.csv
- Shot vs NDCG/HR for CNN vs ViT
- Aggregates by (model, encoder, coldstart_policy, fewshot_k)
Saves plots to data/plots/*.png
"""
import os, argparse, pandas as pd
import matplotlib.pyplot as plt

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def plot_shot_vs_metric(df, metric, out_path, title):
    # one plot per metric: x=fewshot_k, y=metric, grouped by encoder
    plt.figure()
    piv = df.pivot_table(index="fewshot_k", columns="encoder", values=metric, aggfunc="mean")
    piv.sort_index(inplace=True)
    piv.plot(marker="o")  # matplotlib default colors, one chart only
    plt.title(title)
    plt.xlabel("fewshot_k")
    plt.ylabel(metric)
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=str, default="data/results.csv")
    ap.add_argument("--outdir", type=str, default="data/plots")
    ap.add_argument("--model", type=str, default=None, help="filter by model name")
    ap.add_argument("--policy", type=str, default=None, help="filter by coldstart_policy")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    df = pd.read_csv(args.results)
    if args.model:
        df = df[df["model"] == args.model]
    if args.policy:
        df = df[df["coldstart_policy"] == args.policy]

    # Plot NDCG and HR vs fewshot_k
    for metric in ["ndcg_at_10", "hr_at_10"]:
        out = os.path.join(args.outdir, f"{args.model or 'ALL'}_{args.policy or 'ALL'}_{metric}.png")
        plot_shot_vs_metric(df, metric, out, f"{metric} vs fewshot_k ({args.model or 'ALL'}, {args.policy or 'ALL'})")
        print("Saved", out)

if __name__ == "__main__":
    main()
