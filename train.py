import argparse
import os
import csv
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import RHDDatasetCoords
from losses import HeatmapCoordLoss
from architecture import Backbone, Heatmap_reg
from utils import save_checkpoint, EarlyStopper


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
    parser.add_argument("--root", default="data/RHD_published_v2")
    parser.add_argument("--checkpoint-root", default="training_results")
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hand", default="right", choices=["left", "right", "auto"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--beta", type=float, default=100.0)
    parser.add_argument("--train-dataset-length", default="0")
    args = parser.parse_args()

    # print device details
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
        print("Current device:", torch.cuda.current_device())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    job_id = args.job_id if args.job_id is not None else "local"
    run_dir = os.path.join(args.checkpoint_root, str(job_id))
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    
    csv_path = os.path.join(run_dir, "losses.csv")
    csv_exists = os.path.exists(csv_path)

    if not csv_exists:
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "val_loss", "seconds"])

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

    # limit input range
    if int(args.train_dataset_length) > 0:
        N = int(args.train_dataset_length)
        N = min(N, len(train_ds))
        subset_train_ds = Subset(train_ds, range(N))
    else:
         subset_train_ds = train_ds

    train_loader = DataLoader(
        subset_train_ds,
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

    early_stopper = EarlyStopper(patience=5)
    best_val = float("inf")
    for epoch in range(args.epochs):
        t0 = time.time()
        
        # train & val loss
        train_loss = train_one_epoch(model, train_loader, loss_fn, optim, device)
        val_loss = validate(model, val_loader, loss_fn, device)
        
        elapsed = time.time() - t0

        # save loss to csv
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([epoch, float(train_loss), float(val_loss), float(elapsed)])

        # checkpoints
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optim.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        save_checkpoint(ckpt, os.path.join(ckpt_dir, f"epoch_{epoch}.pt"))
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(ckpt, os.path.join(ckpt_dir, "best.pt"))

        # early stopper
        if early_stopper.early_stop(val_loss):             
            print("early stop")
            break


if __name__ == "__main__":
    main()
