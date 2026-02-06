import argparse
import os
import csv

import matplotlib.pyplot as plt


def read_losses(csv_path):
    epochs, train_losses, val_losses = [], [], []
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))
    return epochs, train_losses, val_losses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-root", default="training_results")
    p.add_argument("--job-id", required=True)
    p.add_argument("--csv-name", default="losses.csv")
    p.add_argument("--out-name", default="loss_plot.png")
    args = p.parse_args()

    run_dir = os.path.join(args.checkpoint_root, str(args.job_id))
    csv_path = os.path.join(run_dir, args.csv_name)
    out_path = os.path.join(run_dir, args.out_name)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}")

    epochs, train_losses, val_losses = read_losses(csv_path)

    plt.figure()
    plt.plot(epochs, train_losses, label="train")
    plt.plot(epochs, val_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"Loss curves (job {args.job_id})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
