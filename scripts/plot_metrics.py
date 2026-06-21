import argparse
import json
import os

import matplotlib.pyplot as plt


def plot_single_history(model_type, input_dir, output_dir, prefix=""):
    json_path = os.path.join(input_dir, f"{model_type}_history.json")

    prefix_str = f"{prefix}_" if prefix else ""
    save_path = os.path.join(output_dir, f"{prefix_str}{model_type}_metrics.png")

    if not os.path.exists(json_path):
        print(f"[WARNING] Could not find {json_path}. Skipping individual plot for {model_type.upper()}.")
        return

    with open(json_path, "r") as f:
        history = json.load(f)

    train_history = history["train"]
    val_history = history["val"]
    epochs = range(1, len(val_history["loss"]) + 1)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Machine Translation Training Metrics ({model_type.upper()})", fontsize=16, fontweight="bold", y=1.05)

    def make_plot(ax, train_metric, val_metric, title, ylabel, val_color="tab:orange"):
        if train_metric and train_metric in train_history:
            ax.plot(epochs, train_history[train_metric], label="Train", color="tab:blue", linewidth=2)

        if val_metric and val_metric in val_history:
            ax.plot(epochs, val_history[val_metric], label="Validation", color=val_color, linewidth=2, linestyle="--")

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Epochs", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.7)

    make_plot(axs[0], "loss", "loss", "Cross Entropy Loss", "Loss")
    make_plot(axs[1], "perplexity", "perplexity", "Perplexity", "PPL")
    make_plot(axs[2], None, "bleu", "Validation BLEU Score", "BLEU (%)", val_color="tab:green")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[INFO] Plot saved successfully as '{save_path}'")


def plot_comparative_study(models, input_dir, output_dir, prefix=""):
    prefix_str = f"{prefix}_" if prefix else ""
    save_path = os.path.join(output_dir, f"{prefix_str}models_comparison.png")

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Model Architecture Comparison", fontsize=16, fontweight="bold", y=1.02)

    colors = {"rnn": "tab:red", "lstm": "tab:blue", "gru": "tab:orange", "transformer": "tab:purple"}

    plotted_any = False

    for model in models:
        json_path = os.path.join(input_dir, f"{model}_history.json")
        if not os.path.exists(json_path):
            continue

        with open(json_path, "r") as f:
            history = json.load(f)

        val_history = history["val"]
        epochs = range(1, len(val_history["loss"]) + 1)

        axs[0].plot(epochs, val_history["loss"], label=model.upper(), color=colors.get(model, "black"), linewidth=2)
        axs[1].plot(epochs, val_history["bleu"], label=model.upper(), color=colors.get(model, "black"), linewidth=2)
        plotted_any = True

    if not plotted_any:
        print(f"[WARNING] No history files found in {input_dir}. Skipping comparative plot.")
        return

    axs[0].set_title("Validation Loss Comparison", fontsize=14)
    axs[0].set_xlabel("Epochs", fontsize=12)
    axs[0].set_ylabel("Loss", fontsize=12)
    axs[0].legend()
    axs[0].grid(True, linestyle=":", alpha=0.7)

    axs[1].set_title("Validation BLEU Score Comparison", fontsize=14)
    axs[1].set_xlabel("Epochs", fontsize=12)
    axs[1].set_ylabel("BLEU Score (%)", fontsize=12)
    axs[1].legend()
    axs[1].grid(True, linestyle=":", alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[INFO] Comparative plot saved successfully as '{save_path}'")


def main():
    parser = argparse.ArgumentParser(description="Generate Training Plots for Documentation")
    parser.add_argument("--model", type=str, default="all", choices=["all", "rnn", "lstm", "gru", "transformer"])
    parser.add_argument("--input_dir", type=str, default=".", help="Directory containing the _history.json files.")
    parser.add_argument("--output_dir", type=str, default="docs/figs", help="Directory to save the PNG plots.")
    parser.add_argument("--prefix", type=str, default="", help="Prefix to prepend to filenames (e.g., 'v6')")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    models = ["rnn", "lstm", "gru", "transformer"] if args.model == "all" else [args.model]

    print(f"[INFO] Scanning '{args.input_dir}' for history files...")
    print(f"[INFO] Saving plots to '{args.output_dir}'...")

    for current_model in models:
        plot_single_history(current_model, args.input_dir, args.output_dir, args.prefix)

    if args.model == "all":
        plot_comparative_study(models, args.input_dir, args.output_dir, args.prefix)


if __name__ == "__main__":
    main()
