import argparse
import os

import torch

from src.dataset import build_dataloaders, load_data_lines
from src.infer import run_inference
from src.optimize import run_optimization
from src.tester import run_test
from src.tokenizer import BasicTokenizer
from src.trainer import run_training


def main():
    parser = argparse.ArgumentParser(description="Machine Translation Master Entrypoint")

    parser.add_argument("--mode", type=str, required=True, choices=["test", "train", "optimize", "infer"])
    parser.add_argument("--model", type=str, required=True, choices=["rnn", "lstm", "gru", "transformer"])
    parser.add_argument("--dataset", type=str, default="dataset/tatoeba")
    parser.add_argument("--tokenizer", type=str, default="basic", choices=["basic"])

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument("--embed_size", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)

    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_enc", type=int, default=3)
    parser.add_argument("--num_dec", type=int, default=3)
    parser.add_argument("--dim_ff", type=int, default=1024)
    args = parser.parse_args()

    os.makedirs("artifacts", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.benchmark = True

    if args.tokenizer == "basic":
        src_tok, tgt_tok = BasicTokenizer(), BasicTokenizer()
    else:
        raise NotImplementedError("Other tokenizers are not yet integrated.")

    if args.mode == "infer":
        run_inference(args, src_tok, tgt_tok, device)
        return

    print(f"[INFO] Preparing {args.dataset}...")
    (train_src, train_tgt), (val_src, val_tgt) = load_data_lines(args.dataset)

    if args.mode == "optimize":
        train_src, train_tgt = train_src[:10000], train_tgt[:10000]
        val_src, val_tgt = val_src[:1000], val_tgt[:1000]
    elif args.mode == "test":
        train_src, train_tgt = train_src[:200], train_tgt[:200]
        val_src, val_tgt = val_src[:50], val_tgt[:50]
        args.batch_size = min(args.batch_size, 32)

    src_tok.fit(train_src, max_vocab=15000)
    tgt_tok.fit(train_tgt, max_vocab=15000)

    if args.mode in ["train", "optimize"]:
        src_tok.save_vocab(f"artifacts/src_vocab_{args.tokenizer}.json")
        tgt_tok.save_vocab(f"artifacts/tgt_vocab_{args.tokenizer}.json")

    train_loader, val_loader = build_dataloaders(args, (train_src, train_tgt), (val_src, val_tgt), src_tok, tgt_tok)

    if args.mode == "train":
        run_training(args, train_loader, val_loader, src_tok, tgt_tok, device)
    elif args.mode == "optimize":
        run_optimization(args, train_loader, val_loader, src_tok, tgt_tok, device)
    elif args.mode == "test":
        run_test(args, train_loader, src_tok, tgt_tok, device)


if __name__ == "__main__":
    main()
