import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GROUPS = [
    ("age_group", "Faixa etária"),
    ("gender", "Sexo"),
    ("neighbourhood", "Bairro"),
    ("bolsa_familia", "Bolsa Família"),
    ("hypertension", "Hipertensão"),
    ("diabetes", "Diabetes"),
    ("alcoholism", "Alcoolismo"),
    ("handicap", "Deficiências"),
    ("waiting_group", "Dias de espera"),
    ("weekday", "Dia da semana"),
    ("sms_received", "Recebeu SMS"),
]

CATEGORY_LABELS = {
    "gender": {"F": "Feminino", "M": "Masculino"},
    "weekday": {0: "Seg", 1: "Ter", 2: "Qua", 3: "Qui", 4: "Sex", 5: "Sáb", 6: "Dom"},
    "bolsa_familia": {0: "Não", 1: "Sim"},
    "hypertension": {0: "Não", 1: "Sim"},
    "diabetes": {0: "Não", 1: "Sim"},
    "alcoholism": {0: "Não", 1: "Sim"},
    "sms_received": {0: "Não", 1: "Sim"},
}

WAITING_ORDER = ["0", "1–2", "3–7", "8–14", "15–30", "31–60", "61+"]

STRATEGY_LABELS = {
    "unbalanced": "Sem balanceamento",
    "class_weight": "Peso de classe",
    "undersampling": "Subamostragem",
    "smote": "SMOTENC",
}

COMPARISON_METRICS = [
    ("pr_auc_ap", "AP (precisão média)", False),
    ("recall", "Recall (limiar 0,5)", True),
    ("precision_at_20", "Precisão no top 20%", True),
    ("recall_at_20", "Recall no top 20%", True),
]

COUNT_COLOR = "#4C72B0"
RATE_COLOR = "#DD8452"


def _labels(table, column):
    mapping = CATEGORY_LABELS.get(column, {})
    return [mapping.get(value, str(value)) for value in table[column]]


def _plot_group(table, column, title, path):
    if column == "neighbourhood":
        fig, ax = plt.subplots(figsize=(8, max(6.0, 0.28 * len(table))))
        positions = np.arange(len(table))
        ax.barh(positions, table["n"], color=COUNT_COLOR)
        ax.set_yticks(positions, _labels(table, column))
        ax.invert_yaxis()
        ax.set_xlabel("Agendamentos")
        twin = ax.twiny()
        twin.plot(table["no_show_rate"] * 100, positions, "o-",
                  color=RATE_COLOR, markersize=3, linewidth=1)
        twin.set_xlabel("Taxa de falta (%)")
    else:
        fig, ax = plt.subplots(figsize=(7, 4))
        positions = np.arange(len(table))
        ax.bar(positions, table["n"], color=COUNT_COLOR)
        ax.set_xticks(positions, _labels(table, column))
        ax.set_ylabel("Agendamentos")
        twin = ax.twinx()
        twin.plot(positions, table["no_show_rate"] * 100, "o-", color=RATE_COLOR)
        twin.set_ylabel("Taxa de falta (%)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_distribution(table, column, title, xlabel, path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(table[column], table["n"], color=COUNT_COLOR)
    ax.set(title=title, xlabel=xlabel, ylabel="Agendamentos")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_sms_by_waiting(table, path):
    pivot = table.pivot(index="waiting_group", columns="sms_received",
                        values="no_show_rate")
    order = [group for group in WAITING_ORDER if group in pivot.index]
    pivot = pivot.reindex(order)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    positions = np.arange(len(pivot))
    width = 0.38
    for index, (column, label) in enumerate([(0, "Sem SMS"), (1, "Com SMS")]):
        if column not in pivot.columns:
            continue
        values = pivot[column] * 100
        observed = values.notna()
        ax.bar(positions[observed] + (index - 0.5) * width, values[observed],
               width, label=label)
    ax.set_xticks(positions, pivot.index.astype(str))
    ax.set(title="Taxa de falta por faixa de espera e SMS",
           xlabel="Dias de espera", ylabel="Taxa de falta (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_comparison(table, path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    strategies = [STRATEGY_LABELS.get(name, name) for name in table.index]
    positions = np.arange(len(table))
    for ax, (metric, title, percent) in zip(axes.flat, COMPARISON_METRICS):
        factor = 100.0 if percent else 1.0
        ax.bar(positions, table[f"{metric}_mean"] * factor,
               yerr=table[f"{metric}_std"] * factor, capsize=4,
               color=COUNT_COLOR)
        ax.set_xticks(positions, strategies, rotation=20)
        ax.set_title(title)
        ax.set_ylabel("%" if percent else "AP")
    fig.suptitle("Comparação de estratégias de balanceamento (validação cruzada)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render(output="reports/sprints-3-4"):
    output = Path(output)
    eda_dir = output / "eda"
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    written = []
    for column, title in GROUPS:
        table = pd.read_csv(eda_dir / f"{column}.csv")
        path = figures / f"eda_{column}.png"
        _plot_group(table, column, title, path)
        written.append(path)
    for column, title, xlabel in [
        ("age", "Distribuição de idade", "Idade"),
        ("waiting_days", "Distribuição de dias de espera", "Dias de espera"),
    ]:
        table = pd.read_csv(eda_dir / f"{column}.csv")
        path = figures / f"eda_distribuicao_{column}.png"
        _plot_distribution(table, column, title, xlabel, path)
        written.append(path)
    path = figures / "eda_sms_por_espera.png"
    _plot_sms_by_waiting(pd.read_csv(eda_dir / "sms_by_waiting.csv"), path)
    written.append(path)
    comparison = pd.read_csv(output / "imbalance" / "comparison.csv", index_col=0)
    path = figures / "imbalance_comparacao.png"
    _plot_comparison(comparison, path)
    written.append(path)
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/sprints-3-4")
    args = parser.parse_args()
    for path in render(args.output):
        print(path)
