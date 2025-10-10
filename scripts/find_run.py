#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script a riga di comando per trovare un UNICO MLflow Run ID.

Interroga un repository MLflow locale basandosi su nome dell'esperimento,
parametri e tag. Lo script è progettato per l'automazione:
- In caso di successo (una sola run trovata), stampa solo l'ID della run su stdout
  e termina con codice di uscita 0.
- In caso di fallimento (zero o più di una run trovata), stampa un messaggio
  di errore dettagliato su stderr e termina con codice di uscita 1.
"""
import argparse
import sys
from pathlib import Path

import mlflow
import pandas as pd


def find_unique_run_id(experiment_name: str, params: dict, tags: dict) -> str:
    """
    Trova l'ID di una run unica che matcha i criteri specificati.

    Args:
        experiment_name (str): Il nome dell'esperimento MLflow.
        params (dict): Un dizionario di parametri per filtrare le run.
        tags (dict): Un dizionario di tag per filtrare le run.

    Returns:
        str: L'ID della run trovata.

    Raises:
        ValueError: Se vengono trovate zero o più di una run.
    """
    filter_parts = []
    if params:
        filter_parts.extend([f"params.{k} = '{v}'" for k, v in params.items()])
    if tags:
        filter_parts.extend([f"tags.{k} = '{v}'" for k, v in tags.items()])

    filter_string = " and ".join(filter_parts)

    try:
        runs_df = mlflow.search_runs(
            experiment_names=[experiment_name],
            filter_string=filter_string,
            order_by=["start_time DESC"],
        )
    except mlflow.exceptions.MlflowException as e:
        raise ValueError(f"ERRORE: Impossibile cercare le run: {e}")

    if len(runs_df) == 0:
        raise ValueError(
            f"ERRORE: Nessuna run trovata per l'esperimento '{experiment_name}' "
            f"con i filtri: params={params}, tags={tags}"
        )

    if len(runs_df) == 1:
        return runs_df.iloc[0]["run_id"]

    # --- Logica di formattazione per il caso di ambiguità (> 1 run) ---

    # Imposta le opzioni di Pandas per un output pulito e non troncato
    pd.set_option("display.width", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", None)

    # Identifica le colonne dei parametri che hanno valori diversi tra le run
    param_cols = sorted([c for c in runs_df.columns if c.startswith("params.")])
    differing_param_cols = [col for col in param_cols if runs_df[col].nunique() > 1]

    # Colonne da mostrare sempre + quelle che differiscono per aiutare l'utente
    display_cols = ["run_id", "start_time"] + differing_param_cols

    # Pulisce i nomi delle colonne (rimuove 'params.') per la visualizzazione
    display_df = runs_df[display_cols].rename(
        columns=lambda c: c.replace("params.", "")
    )

    error_message = (
        f"ERRORE: La query è ambigua. Trovate {len(runs_df)} run.\n"
        f"Aggiungi filtri --param per i parametri che differiscono per ottenere un risultato unico.\n\n"
        "Run Trovate (mostrando solo le colonne chiave e i parametri differenti):\n"
    )

    run_info_table = display_df.to_string(index=False)

    raise ValueError(error_message + run_info_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trova un UNICO MLflow Run ID. Fallisce se la query è ambigua.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--experiment-name", required=True, help="Nome dell'esperimento MLflow."
    )
    parser.add_argument(
        "--param",
        action="append",
        help="Parametro nel formato 'key=value'. Può essere specificato più volte.",
    )
    parser.add_argument(
        "--tag",
        action="append",
        help="Tag nel formato 'key=value'. Può essere specificato più volte.",
    )

    args = parser.parse_args()

    # --- Setup dell'ambiente ---
    try:
        # Trova la root del progetto (la cartella sopra 'scripts/')
        project_root = Path(__file__).resolve().parent.parent
        mlflow_tracking_uri = project_root / "mlruns"

        if not mlflow_tracking_uri.exists():
            raise FileNotFoundError(f"Directory 'mlruns' non trovata in {project_root}")

        mlflow.set_tracking_uri(mlflow_tracking_uri.as_uri())
    except Exception as e:
        print(
            f"ERRORE CRITICO: Impossibile configurare l'ambiente MLflow.",
            file=sys.stderr,
        )
        print(f"Dettagli: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Esecuzione della logica principale ---
    try:
        parsed_params = dict(p.split("=", 1) for p in args.param) if args.param else {}
        parsed_tags = dict(t.split("=", 1) for t in args.tag) if args.tag else {}

        if not parsed_params and not parsed_tags:
            print(
                "ERRORE: Specificare almeno un --param o un --tag per filtrare le run.",
                file=sys.stderr,
            )
            parser.print_help(sys.stderr)
            sys.exit(1)

        run_id = find_unique_run_id(args.experiment_name, parsed_params, parsed_tags)

        # In caso di successo, stampa solo l'ID su stdout
        print(run_id)

    except ValueError as e:
        # In caso di fallimento, stampa il messaggio informativo su stderr
        print(e, file=sys.stderr)
        sys.exit(1)
