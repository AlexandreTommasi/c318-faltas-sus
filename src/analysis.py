"""Análise reprodutível das sprints 2–3: python -m src.analysis."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from src.data import load_raw, clean
from src.features import build_features, add_patient_history, NUMERIC_FEATURES, HISTORY_FEATURES
from src.models import split_by_patient, split_temporal, run_baselines


def rate_table(df, column):
    """Suporte, faltas e taxa juntos para contextualizar grupos pequenos."""
    return df.groupby(column, observed=True, dropna=False)['no_show'].agg(
        appointments='size', no_shows='sum', rate='mean').reset_index()


def run(raw_path, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    df = add_patient_history(build_features(clean(load_raw(raw_path))))
    train, test = split_by_patient(df)
    temporal_train, temporal_test = split_temporal(df)
    results = {
        'patient_split': run_baselines(df, train, test),
        'temporal_split': run_baselines(df, temporal_train, temporal_test),
        'metadata': {'rows': len(df), 'patients': int(df.patient_id.nunique()),
                     'train_rows': len(train), 'test_rows': len(test),
                     'temporal_train_rows': len(temporal_train), 'temporal_test_rows': len(temporal_test),
                     'seed': 42, 'temporal_cutoff': '2016-05-25',
                     'pr_auc_definition': 'average_precision_score (AP)',
                     'history_coverage': float(df.previous_appointments.gt(0).mean())},
    }
    (output / 'baselines.json').write_text(json.dumps(results, indent=2) + '\n')
    # EDA e seleção no treino: teste reservado para avaliação.
    eda = df.iloc[train].copy()
    eda['age_band'] = pd.cut(eda.age, [-1, 5, 12, 18, 35, 60, 120])
    eda['waiting_band'] = pd.cut(eda.waiting_days, [-1, 0, 2, 7, 14, 30, 60, 365])
    eda['has_history'] = eda.previous_appointments.gt(0)
    columns = ['age_band', 'gender', 'neighbourhood', 'bolsa_familia', 'hypertension',
               'diabetes', 'alcoholism', 'handicap', 'waiting_band', 'weekday', 'has_history']
    for col in columns:
        table = rate_table(eda, col)
        table.to_csv(output / f'rates_{col}.csv', index=False)
        fig, ax = plt.subplots(figsize=(12 if col == 'neighbourhood' else 8, 4))
        ax.bar(table[col].astype(str), table.rate)
        ax.set(xlabel=col, ylabel='Taxa de falta', title='EDA no treino por paciente')
        ax.tick_params(axis='x', rotation=90 if col == 'neighbourhood' else 35)
        fig.tight_layout()
        fig.savefig(output / f'rates_{col}.png', dpi=140)
        plt.close(fig)
    eda.groupby(['waiting_band', 'sms_received'], observed=True).no_show.agg(
        appointments='size', no_shows='sum', rate='mean').to_csv(output / 'sms_by_waiting.csv')
    eda.groupby(['has_history', 'previous_no_shows'], observed=True).no_show.agg(
        appointments='size', rate='mean').to_csv(output / 'history_rates.csv')
    corr = eda[NUMERIC_FEATURES + HISTORY_FEATURES + ['no_show']].corr(method='spearman')
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
