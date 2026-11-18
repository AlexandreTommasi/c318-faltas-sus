"""Testes unitários para add_patient_history."""
import unittest

import pandas as pd

from src.features import add_patient_history


def _df(*rows):
    """Cria DataFrame de teste: (patient_id, scheduled_day, appointment_day, no_show)."""
    return pd.DataFrame({
        "patient_id": [r[0] for r in rows],
        "scheduled_day": pd.to_datetime([r[1] for r in rows], utc=True),
        "appointment_day": pd.to_datetime([r[2] for r in rows], utc=True),
        "no_show": [r[3] for r in rows],
    })


class HistoryTests(unittest.TestCase):

    def test_future_appointment_does_not_count(self):
        """Consulta futura do mesmo paciente não entra no histórico."""
        # Ambas as consultas têm appointment_day no futuro em relação ao scheduled da outra.
        df = _df(
            ("A", "2016-05-02", "2016-05-10", 1),  # scheduled 05-02; appt_day 05-10
            ("A", "2016-05-05", "2016-05-07", 0),  # scheduled 05-05; appt_day da outra (05-10) > 05-05
        )
        result = add_patient_history(df)
        # Nenhuma tem appointment_day anterior ao scheduled_day da outra → histórico zerado
        self.assertEqual(result.loc[0, "previous_appointments"], 0)
        self.assertEqual(result.loc[1, "previous_appointments"], 0)

    def test_same_day_does_not_count(self):
        """Consulta com appointment_day igual ao scheduled_day atual não conta."""
        df = _df(
            ("A", "2016-05-01", "2016-05-03", 1),  # appointment_day 05-03
            ("A", "2016-05-03", "2016-05-05", 0),  # scheduled 05-03; appt_day acima == scheduled → não conta
        )
        result = add_patient_history(df)
        self.assertEqual(result.loc[1, "previous_appointments"], 0)
        self.assertEqual(result.loc[1, "has_history"], 0)

    def test_past_appointment_counts_correctly(self):
        """Consulta anterior conta com falta e taxa registradas corretamente."""
        df = _df(
            ("A", "2016-05-01", "2016-05-02", 1),  # appointment_day 05-02, faltou
            ("A", "2016-05-04", "2016-05-05", 0),  # scheduled 05-04; 05-02 < 05-04 → conta
        )
        result = add_patient_history(df)
        self.assertEqual(result.loc[1, "previous_appointments"], 1)
        self.assertEqual(result.loc[1, "previous_no_shows"], 1)
        self.assertAlmostEqual(result.loc[1, "previous_no_show_rate"], 1.0)
        self.assertEqual(result.loc[1, "has_history"], 1)

    def test_history_with_no_misses(self):
        """Com histórico mas sem faltas, taxa = 0.0 e has_history = 1."""
        df = _df(
            ("A", "2016-05-01", "2016-05-02", 0),  # compareceu
            ("A", "2016-05-04", "2016-05-05", 0),
        )
        result = add_patient_history(df)
        self.assertEqual(result.loc[1, "previous_appointments"], 1)
        self.assertEqual(result.loc[1, "previous_no_shows"], 0)
        self.assertAlmostEqual(result.loc[1, "previous_no_show_rate"], 0.0)
        self.assertEqual(result.loc[1, "has_history"], 1)

    def test_patients_do_not_mix(self):
        """Histórico de paciente A não contamina o de paciente B."""
        df = _df(
            ("A", "2016-05-01", "2016-05-02", 1),
            ("B", "2016-05-01", "2016-05-02", 1),
            ("B", "2016-05-04", "2016-05-05", 0),
        )
        result = add_patient_history(df)
        # B segunda consulta: só conta a primeira de B (não a de A)
        self.assertEqual(result.loc[2, "previous_appointments"], 1)

    def test_order_and_index_independence(self):
        """Resultado é o mesmo com linhas fora de ordem e índice não sequencial."""
        df = _df(
            ("A", "2016-05-01", "2016-05-02", 1),  # idx 0 → sem histórico
            ("A", "2016-05-04", "2016-05-05", 0),  # idx 1 → 1 consulta anterior
        )
        # Embaralha: inverte as linhas e atribui índice externo [10, 20]
        df_rev = df.iloc[::-1].copy()
        df_rev.index = [10, 20]  # idx 10 = conteúdo original idx 1, idx 20 = idx 0

        r = add_patient_history(df)
        r_rev = add_patient_history(df_rev)

        # Linha com scheduled 05-01: sem histórico (idx 0 no original, idx 20 no reverso)
        self.assertEqual(r.loc[0, "previous_appointments"], 0)
        self.assertEqual(r_rev.loc[20, "previous_appointments"], 0)
        # Linha com scheduled 05-04: 1 consulta anterior (idx 1 no original, idx 10 no reverso)
        self.assertEqual(r.loc[1, "previous_appointments"], 1)
        self.assertEqual(r_rev.loc[10, "previous_appointments"], 1)

    def test_no_history_gives_zeros(self):
        """Paciente com uma única consulta tem tudo zerado e has_history = 0."""
        df = _df(("X", "2016-05-01", "2016-05-02", 0))
        result = add_patient_history(df)
        self.assertEqual(result.loc[0, "previous_appointments"], 0)
        self.assertEqual(result.loc[0, "previous_no_shows"], 0)
        self.assertAlmostEqual(result.loc[0, "previous_no_show_rate"], 0.0)
        self.assertEqual(result.loc[0, "has_history"], 0)


if __name__ == "__main__":
    unittest.main()
