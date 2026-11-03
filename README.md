# Machine Translation in PyTorch

A neural machine translation system written in plain PyTorch, used as a
comparative study. It trains four architectures (RNN, LSTM, GRU, Transformer)
against three tokenization strategies (word-level, custom BPE, SentencePiece)
on the Tatoeba English-French corpus of roughly 230k sentence pairs, and
records where each combination succeeds and where it does not.

![Architecture comparison](docs/figs/architectures_comparison.png)

## What it does

- Trains four encoder-decoder architectures behind one interface, so a run
  differs only by `--model`.
- Implements three tokenizers from scratch to measure the out-of-vocabulary
  problem directly: regex word-level, a pure-Python BPE, and a SentencePiece
  unigram model.
- Runs Optuna searches over architectural shape (embedding size, layer counts,
  feed-forward width) while leaving training hyperparameters fixed.
- Checkpoints model, optimizer, metric history, and early-stopping counters
  every epoch. `--resume` reloads the run's vocabulary alongside the weights,
  since mismatched vocabularies silently corrupt the embedding rows.
- Reports BLEU through `torchmetrics` and plots the comparisons in `docs/figs`.

## Samples

Seven configurations are recorded side by side in
[docs/samples.md](docs/samples.md). The word-level RNN repeats the same
sentence for every prompt, the custom BPE model emits compressed fragments
(`peu t m voi`), and the SentencePiece Transformer produces complete French
sentences. None of these models is usable outside the study.

Experiment notes and the optimisation history are in
[docs/experiments.md](docs/experiments.md).

## Run it

```bash
git clone https://github.com/abderrahmenex86/translation
cd translation
virtualenv venv --python python3.12
source venv/bin/activate
pip install -r requirements.txt

python tools.py --mode download                          # fetch Tatoeba
python tools.py --mode verify --tokenizer bpe --vocab_size 2000
python main.py --mode test --model transformer --tokenizer spm
```

Python 3.10+ and a CUDA GPU. The verify and test modes run without training
and are the fastest way to check an install.

## Training

```bash
# Optuna sweep over model shape, written to artifacts/<run>/optuna_summary.json
python main.py --mode optimize --model transformer --tokenizer spm

# Train with an explicit configuration; stops early on val loss
python main.py --mode train --model transformer --tokenizer spm \
  --d_model 256 --dim_ff 1024 --num_enc 3 --num_dec 3 --epochs 50 --patience 7

# Resume the latest run, vocabulary included
python main.py --mode train --model transformer --resume

# Translate interactively
python main.py --mode infer --model transformer
```

## How it works

`main.py` dispatches train, test, optimize, and infer. `tools.py` handles
dataset download, dataloader verification, and plotting. Model construction
lives in `src/factory.py`, so the tokenizer and architecture are chosen at the
CLI and assembled in one place. See [docs/architecture.md](docs/architecture.md)
for the module map and the attention and positional-encoding details.

## Stack

Python 3.12 · PyTorch · torchmetrics (BLEU) · SentencePiece · Optuna ·
Matplotlib · TQDM
