import argparse
import os
import time

import torch

from helpers import translate_sentence
from tokenizer import BasicTokenizer
from train import filter_pairs, get_model, read_text_file


def get_tokenizers():
    src_tok = BasicTokenizer(min_freq=2)
    tgt_tok = BasicTokenizer(min_freq=2)

    if os.path.exists("src_vocab.json") and os.path.exists("tgt_vocab.json"):
        print("-> Loading tokenizers.")
        src_tok.load_vocab("src_vocab.json")
        tgt_tok.load_vocab("tgt_vocab.json")
    else:
        print("-> Rebuilding from dataset.")
        raw_src = read_text_file("dataset/tatoeba/train.en")
        raw_tgt = read_text_file("dataset/tatoeba/train.fr")
        raw_src, raw_tgt = filter_pairs(raw_src, raw_tgt, max_len=50)

        src_tok.fit(raw_src, max_vocab=15000)
        tgt_tok.fit(raw_tgt, max_vocab=15000)

        src_tok.save_vocab("src_vocab.json")
        tgt_tok.save_vocab("tgt_vocab.json")
        print("-> Vocabularies saved to JSON.")

    print(f"-> English Vocab Size: {len(src_tok)}")
    print(f"-> French Vocab Size: {len(tgt_tok)}")
    return src_tok, tgt_tok


def main():
    parser = argparse.ArgumentParser(description="Machine Translation CLI")
    parser.add_argument(
        "--model",
        type=str,
        default="transformer",
        choices=["rnn", "lstm", "gru", "transformer"],
        help="Type of model architecture to load.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device.type.upper()}")

    src_tokenizer, tgt_tokenizer = get_tokenizers()

    model = get_model(
        args.model,
        src_vocab_size=len(src_tokenizer),
        tgt_vocab_size=len(tgt_tokenizer),
        pad_idx=tgt_tokenizer.PAD,
        device=device,
    )

    weights_path = f"best_model_{args.model}.pth"
    print(f"Loading weights from: {weights_path}")

    if not os.path.exists(weights_path):
        print(f"❌ Error: Could not find '{weights_path}'.")
        return

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    print("\n" + "=" * 50)
    print(f"Translation CLI ({args.model.upper()})")
    print("  Type 'q' or 'quit' to exit.")
    print("=" * 50 + "\n")

    while True:
        sentence = input("English > ")

        if sentence.lower().strip() in ["q", "quit", "exit"]:
            print("Exiting...")
            break

        if not sentence.strip():
            continue

        start_time = time.perf_counter()

        try:
            translation = translate_sentence(
                sentence=sentence,
                model=model,
                src_tokenizer=src_tokenizer,
                tgt_tokenizer=tgt_tokenizer,
                device=device,
                model_type=args.model,
            )

            elapsed = (time.perf_counter() - start_time) * 1000

            print(f"French  > {translation}")
            print(f"Latency: {elapsed:.2f} ms\n")

        except Exception as e:
            print(f"An error occurred: {e}\n")


if __name__ == "__main__":
    main()
