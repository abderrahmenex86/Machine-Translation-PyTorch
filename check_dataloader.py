import torch
from torch.utils.data import DataLoader

from dataset import TranslationDataset, collate_fn
from tokenizer import BasicTokenizer
from train import filter_pairs, read_text_file


def inspect_dataloader():
    print("\n" + "=" * 50)
    print("Dataloader Verification")
    print("=" * 50)

    raw_src = read_text_file("dataset/multi30k/train.en")[:1000]
    raw_tgt = read_text_file("dataset/multi30k/train.fr")[:1000]
    raw_src, raw_tgt = filter_pairs(raw_src, raw_tgt, max_len=50)

    src_tokenizer = BasicTokenizer(min_freq=2)
    tgt_tokenizer = BasicTokenizer(min_freq=2)
    src_tokenizer.fit(raw_src)
    tgt_tokenizer.fit(raw_tgt)

    dataset = TranslationDataset(raw_src, raw_tgt, src_tokenizer, tgt_tokenizer)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    src_batch, tgt_batch = next(iter(loader))

    print("\n[BATCH SHAPES]")
    print(f"Source Batch Shape: {src_batch.shape} -> [batch_size, src_sequence_length]")
    print(f"Target Batch Shape: {tgt_batch.shape} -> [batch_size, tgt_sequence_length]")

    print("\n" + "=" * 50)
    print("Sample 0 in batch")
    print("=" * 50)

    src_sample = src_batch[0]
    tgt_sample = tgt_batch[0]

    def decode_with_specials(tensor, tokenizer):
        return " ".join([tokenizer.idx2word.get(idx.item(), f"<UNK:{idx.item()}>") for idx in tensor])

    print("\n[SOURCE (English)]")
    print(f"Raw Tensor: {src_sample.tolist()}")
    print(f"Decoded:    {decode_with_specials(src_sample, src_tokenizer)}")

    print("\n[TARGET (French)]")
    print(f"Raw Tensor: {tgt_sample.tolist()}")
    print(f"Decoded:    {decode_with_specials(tgt_sample, tgt_tokenizer)}")


if __name__ == "__main__":
    inspect_dataloader()
