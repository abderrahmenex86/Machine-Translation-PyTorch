import json
import os

import optuna
import torch
import torch.nn as nn

from src.factory import build_model
from src.trainer import evaluate, train_epoch


def run_optimization(args, train_loader, val_loader, src_tok, tgt_tok, device, run_dir):  # Added run_dir here
    def objective(trial):
        model = build_model(args.model, len(src_tok), len(tgt_tok), tgt_tok.PAD, device, trial, **vars(args))

        lr = 1e-4 if args.model == "transformer" else 5e-3
        criterion = nn.CrossEntropyLoss(ignore_index=tgt_tok.PAD, label_smoothing=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        for epoch in range(3):
            train_epoch(model, train_loader, criterion, optimizer, device, args.model)
            v_loss, _ = evaluate(model, val_loader, criterion, device, args.model, tgt_tok)

            trial.report(v_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return v_loss

    print(f"\n[INFO] Optimizing Architecture for {args.model.upper()}...")
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=15)

    print("\n[SUCCESS] Architectural Search Complete")
    print(f"Best Loss: {study.best_value:.4f}")
    print(f"Best Hyperparameters: {study.best_params}")

    summary_data = {
        "model": args.model,
        "best_loss": study.best_value,
        "best_params": study.best_params,
        "all_trials": [
            {"trial_number": t.number, "params": t.params, "loss": t.value, "state": str(t.state).split(".")[-1]}
            for t in study.trials
        ],
    }

    summary_path = os.path.join(run_dir, "optuna_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=4)

    print(f"[INFO] Optimization logs and parameters logged to {summary_path}")
