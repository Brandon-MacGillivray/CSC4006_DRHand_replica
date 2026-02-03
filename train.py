import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import RHDDatasetCoords
from losses import HeatmapCoordLoss
from architecture import Backbone, Heatmap_reg
from utils import save_checkpoint


class HandPoseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.head = Heatmap_reg()

    def forward(self, x):
        return self.head(self.backbone(x))


def train_one_epoch(model, loader, loss_fn, optim, device):
    model.train()
    total = 0.0
    for imgs, coords, vis in loader:
        imgs = imgs.to(device)
        coords = coords.to(device)
        vis = vis.to(device)

        pred_hm = model(imgs)
        loss = loss_fn(pred_hm, coords, vis)

        optim.zero_grad()
        loss.backward()
        optim.step()

        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    total = 0.0
    for imgs, coords, vis in loader:
        imgs = imgs.to(device)
        coords = coords.to(device)
        vis = vis.to(device)

        pred_hm = model(imgs)
        loss = loss_fn(pred_hm, coords, vis)
        total += loss.item()
    return total / max(len(loader), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/RHD_v1-1")
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hand", default="right", choices=["left", "right", "auto"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--beta", type=float, default=100.0)
    parser.add_argument("--checkpoint-dir", default="mk_3/checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = RHDDatasetCoords(
        args.root,
        split="training",
        input_size=args.input_size,
        hand=args.hand,
        normalize=True,
    )
    val_ds = RHDDatasetCoords(
        args.root,
        split="evaluation",
        input_size=args.input_size,
        hand=args.hand,
        normalize=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = HandPoseNet().to(device)
    loss_fn = HeatmapCoordLoss(beta=args.beta, normalize=True)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optim, device)
        val_loss = validate(model, val_loader, loss_fn, device)
        print(f"epoch {epoch}: train {train_loss:.4f} val {val_loss:.4f}")

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optim.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        save_checkpoint(ckpt, os.path.join(args.checkpoint_dir, f"epoch_{epoch}.pt"))
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(ckpt, os.path.join(args.checkpoint_dir, "best.pt"))


if __name__ == "__main__":
    main()
