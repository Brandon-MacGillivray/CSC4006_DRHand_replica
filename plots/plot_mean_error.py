import argparse
import csv
from collections import defaultdict

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="mk_3/train_subset_mean.csv")
    parser.add_argument("--sample-id", type=int, default=-1)
    args = parser.parse_args()

    data = defaultdict(list)
    epochs = set()

    with open(args.csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = int(row["epoch"])
            sample_id = int(row["sample_id"])
            mean_err = float(row["mean_err"])
            data[sample_id].append((epoch, mean_err))
            epochs.add(epoch)

    if args.sample_id >= 0:
        if args.sample_id not in data:
            raise ValueError(f"sample-id {args.sample_id} not found in CSV")
        items = [args.sample_id]
    else:
        items = sorted(data.keys())

    plt.figure(figsize=(7, 4))
    for sample_id in items:
        series = sorted(data[sample_id], key=lambda x: x[0])
        xs = [e for e, _ in series]
        ys = [v for _, v in series]
        plt.plot(xs, ys, label=str(sample_id))

    plt.xlabel("Epoch")
    plt.ylabel("Mean Error (normalized)")
    plt.title("Mean Error per Sample ID")
    if len(items) <= 10:
        plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
