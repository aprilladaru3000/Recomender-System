"""
Distill ViT visual embeddings (teacher) into a lightweight CNN (student).
- Loads items.csv (image_path) and teacher embeddings NPZ (keyed by item_id as string).
- Trains a ResNet18 student to regress to teacher vectors (MSE).
- Saves student weights and precomputed embeddings NPZ for use as --precomputed.
"""
import os, argparse, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

class ImgDataset(Dataset):
    def __init__(self, items_df, data_root, teacher_npz, out_dim=256):
        self.df = items_df.reset_index(drop=True)
        self.root = data_root
        self.teacher = dict(teacher_npz)  # str(item_id)->vec
        self.out_dim = out_dim
        self.tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        item_id = str(r["item_id"])
        path = r["image_path"]
        img = Image.open(os.path.join(self.root, path)).convert("RGB")
        x = self.tf(img)
        y = torch.tensor(self.teacher[item_id], dtype=torch.float32)
        return x, y, item_id

class StudentCNN(nn.Module):
    def __init__(self, out_dim=256, pretrained=True):
        super().__init__()
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()
        self.backbone = m
        self.head = nn.Linear(feat_dim, out_dim)

    def forward(self, x):
        f = self.backbone(x)
        return self.head(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--teacher-npz", type=str, required=True, help="NPZ keyed by item_id")
    ap.add_argument("--out-weights", type=str, default="data/student_resnet18.pth")
    ap.add_argument("--out-emb", type=str, default="data/student_emb.npz")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    items = pd.read_csv(os.path.join(args.data_root, "items.csv"))
    # keep only items present in teacher npz
    teacher_npz = np.load(args.teacher_npz, allow_pickle=True)
    item_ids = set(teacher_npz.files)
    items = items[items["item_id"].astype(str).isin(item_ids)].reset_index(drop=True)

    ds = ImgDataset(items, args.data_root, teacher_npz)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = StudentCNN(out_dim=teacher_npz[teacher_npz.files[0]].shape[-1]).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    model.train()
    for ep in range(1, args.epochs+1):
        total = 0.0
        for x, y, _ in tqdm(dl, desc=f"Distill ep{ep}"):
            x = x.to(args.device); y = y.to(args.device)
            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        print(f"[ep {ep}] distill loss={total/len(dl):.4f}")

    # save weights
    os.makedirs(os.path.dirname(args.out_weights), exist_ok=True)
    torch.save(model.state_dict(), args.out_weights)
    print("Saved student weights to", args.out_weights)

    # precompute student embeddings NPZ to use as --precomputed
    model.eval()
    emb = {}
    with torch.no_grad():
        for x, _, item_id in tqdm(DataLoader(ds, batch_size=32, shuffle=False), desc="Precompute student emb"):
            x = x.to(args.device)
            v = model(x).cpu().numpy()
            for i, iid in enumerate(item_id):
                emb[iid] = v[i]
    np.savez_compressed(args.out_emb, **emb)
    print("Saved student embeddings to", args.out_emb)

if __name__ == "__main__":
    main()
