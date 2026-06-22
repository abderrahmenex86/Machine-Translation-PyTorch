import argparse
import os

from src.helpers import download_dataset, plot_metrics
from src.verify import run_verification


def main():
    parser = argparse.ArgumentParser(description="Machine Translation Utility Tools")

    parser.add_argument("--mode", type=str, required=True, choices=["download", "verify", "plot"])
    parser.add_argument("--model", type=str, default="all", choices=["all", "rnn", "lstm", "gru", "transformer"])
    parser.add_argument("--dataset", type=str, default="dataset/tatoeba")
    args = parser.parse_args()

    os.makedirs("artifacts", exist_ok=True)

    if args.mode == "download":
        download_dataset(args.dataset)

    elif args.mode == "verify":
        args.batch_size = 4
        args.tokenizer = "basic"
        run_verification(args)

    elif args.mode == "plot":
        plot_metrics(args.model)


if __name__ == "__main__":
    main()
