"""Análise reprodutível das sprints 2–3: python -m src.analysis."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.data import load_raw, clean
from src.features import build_features, NUMERIC_FEATURES
from src.models import split_by_patient, split_temporal, run_baselines


def run(raw_path, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    df = build_features(clean(load_raw(raw_path)))
    train, test = split_by_patient(df)
    temporal_train, temporal_test = split_temporal(df)
    results = {
        'patient_split': run_baselines(df, train, test),
        'temporal_split': run_baselines(df, temporal_train, temporal_test),
        'metadata': {'rows': len(df), 'patients': int(df.patient_id.nunique()),
                     'train_rows': len(train), 'test_rows': len(test),
                     'temporal_train_rows': len(temporal_train), 'temporal_test_rows': len(temporal_test),
                     'seed': 42, 'temporal_cutoff': '2016-05-25',
                     'pr_auc_definition': 'average_precision_score (AP)'},
    }
    (output / 'baselines.json').write_text(json.dumps(results, indent=2) + '\n')
    # Correlações e seleção no treino: teste reservado para avaliação.
    eda = df.iloc[train].copy()
    corr = eda[NUMERIC_FEATURES + ['no_show']].corr(method='spearman')
    corr.to_csv(output / 'spearman.csv')
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90)
    ax.set_yticks(range(len(corr)), corr.index)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output / 'spearman.png', dpi=140)
    plt.close(fig)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', default='data/raw/KaggleV2-May-2016.csv')
    parser.add_argument('--output', default='reports/sprints-2-3')
    args = parser.parse_args()
    print(json.dumps(run(args.raw, args.output), indent=2))
