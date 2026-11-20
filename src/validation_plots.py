"""Render sprint 5 figures from frozen aggregate artifacts only."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.sprint_plots import STRATEGY_LABELS


def render(base="reports/sprint-5"):
    base = Path(base)
    source = base / "evaluation"
    output = base / "figures"
    metadata = json.loads((source / "metadata.json").read_text())
    summary = pd.read_csv(source / "comparison.csv", index_col="strategy")
    roc = pd.read_csv(source / "roc_curves.csv")
    pr = pd.read_csv(source / "precision_recall_curves.csv")
    matrices = pd.read_csv(source / "confusion_matrices.csv")
    output.mkdir(parents=True, exist_ok=True)
    written = []
    for kind, data, x, y, xlabel, ylabel, metric in (
        ("roc", roc, "fpr", "tpr", "Taxa de falsos positivos", "Recall (sensibilidade)", "roc_auc"),
        ("precision_recall", pr, "recall", "precision", "Recall", "Precisao", "pr_auc_ap"),
    ):
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
        for ax, strategy in zip(axes.flat, summary.index):
            for fold, curve in data[data["strategy"] == strategy].groupby("fold", sort=True):
                ax.plot(curve[x], curve[y], linewidth=1.3, alpha=0.8, label=f"Fold {fold}")
                if kind == "precision_recall":
                    prevalence = next(
                        item["validation_prevalence"]
                        for item in metadata["folds"] if item["fold"] == fold
                    )
                    ax.axhline(prevalence, color="gray", linestyle=":", linewidth=0.7)
            if kind == "roc":
                ax.plot([0, 1], [0, 1], "--", color="gray", label="Aleatorio")
            else:
                ax.plot([], [], ":", color="gray", label="Prevalencia por fold")
            mean = summary.loc[strategy, metric + "_mean"]
            std = summary.loc[strategy, metric + "_std"]
            name = "ROC-AUC" if kind == "roc" else "AP"
            ax.set(title=f"{STRATEGY_LABELS[strategy]} | {name} {mean:.3f} (DP {std:.3f})",
                   xlabel=xlabel, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1.02))
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        fig.suptitle("Validacao por paciente no treino | teste reservado\n"
                     "Curvas por fold; pontos reduzidos para exibicao; metricas exatas")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        path = output / f"{kind}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    vmax = matrices["count"].max()
    for ax, strategy in zip(axes.flat, summary.index):
        matrix = (
            matrices[matrices["strategy"] == strategy]
            .pivot(index="actual", columns="predicted", values="count")
            .reindex(index=[0, 1], columns=[0, 1]).to_numpy()
        )
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax)
        for actual, predicted in np.ndindex(2, 2):
            ax.text(predicted, actual, f"{matrix[actual, predicted]:,}", ha="center", va="center",
                    color="white" if matrix[actual, predicted] > vmax / 2 else "black")
        ax.set(title=STRATEGY_LABELS[strategy], xlabel="Previsto", ylabel="Real",
               xticks=[0, 1], yticks=[0, 1], xticklabels=["Compareceu (0)", "Faltou (1)"],
               yticklabels=["Compareceu (0)", "Faltou (1)"])
    fig.suptitle("Matrizes de confusao | limiar fixo 0,5\n"
                 "Soma das validacoes: cada agendamento de treino contado uma vez; teste reservado")
    fig.subplots_adjust(wspace=0.45, hspace=0.4, top=0.87, right=0.8)
    colorbar_axis = fig.add_axes([0.87, 0.18, 0.02, 0.60])
    fig.colorbar(image, cax=colorbar_axis, label="Agendamentos")
    path = output / "confusion_matrices.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="reports/sprint-5")
    args = parser.parse_args()
    for path in render(args.base):
        print(path)
