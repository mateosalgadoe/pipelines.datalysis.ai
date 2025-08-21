import os
import json
from datetime import datetime
import psycopg2
import psycopg2.extras
import pandas as pd
import numpy as np

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


def _env(n, d=None):
    v = os.getenv(n, d)
    return v if v is not None and str(v).strip() != '' else None


def _json_default(o):
    """Convierte tipos no-JSON (Timestamp, numpy, etc.) a algo serializable."""
    if isinstance(o, (pd.Timestamp, datetime)):
        # ISO simple; si prefieres UTC marcada: o.astimezone(timezone.utc).isoformat()
        return o.isoformat()
    if o is pd.NaT:
        return None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    # último recurso: string
    return str(o)


@data_exporter
def export_data(df, *args, **kwargs):
    """
    Inserta UNA FILA POR LEAD en <schema>.api_rocknblock_leads:
      - extraction_ts (timestamp) -> mismo valor para todo el batch
      - request_payload (jsonb)   -> parámetros usados
      - payload (jsonb)           -> lead individual (sin columna extraction_ts)
    """
    if df is None or df.empty:
        print('DF vacío, nada que exportar.')
        return

    schema = _env('ANALYTICS_POSTGRES_SCHEMA', 'raw')
    table  = kwargs.get('table', 'api_rocknblock_leads')

    # timestamp del batch (si el loader lo puso, úsalo; si no, genéralo)
    if 'extraction_ts' in df.columns and not df['extraction_ts'].isna().all():
        run_ts = df['extraction_ts'].iloc[0]
        if isinstance(run_ts, str):
            run_ts = datetime.fromisoformat(run_ts)
    else:
        run_ts = datetime.utcnow()

    request_payload = {
        "endpoint": "/get-leads",
        "base_url": _env('ROCKN_API_BASE_URL'),
        "params": {
            "per_page": int(_env('ROCKN_API_PER_PAGE', '100')),
            "date_from": _env('ROCKN_DATE_FROM'),
            "date_to": _env('ROCKN_DATE_TO'),
            "source": _env('ROCKN_SOURCE'),
            "state": _env('ROCKN_STATE'),
            "state_location": _env('ROCKN_STATE_LOCATION'),
            "city": _env('ROCKN_CITY'),
            "start_page": int(_env('ROCKN_API_START_PAGE', '1')),
            "max_pages": int(_env('ROCKN_API_MAX_PAGES', '0')),
        },
        "count_rows": int(len(df)),
    }
    req_json = json.dumps(request_payload, ensure_ascii=False, default=_json_default)

    conn = psycopg2.connect(
        host=_env("ANALYTICS_POSTGRES_HOST"),
        dbname=_env("ANALYTICS_POSTGRES_DB"),
        user=_env("ANALYTICS_POSTGRES_USER"),
        password=_env("ANALYTICS_POSTGRES_PASSWORD"),
        port=int(_env("ANALYTICS_POSTGRES_PORT", "5432")),
        sslmode=_env("ANALYTICS_POSTGRES_SSLMODE", "require"),
    )
    cur = conn.cursor()

    # asegurar schema/tabla
    cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}";')
    cur.execute(f'''
        CREATE TABLE IF NOT EXISTS "{schema}".{table} (
            id BIGSERIAL PRIMARY KEY,
            extraction_ts TIMESTAMP NOT NULL DEFAULT (now() AT TIME ZONE 'UTC'),
            request_payload JSONB,
            payload JSONB
        );
    ''')

    # preparar filas: quitar extraction_ts del payload
    records = df.to_dict(orient='records')
    rows = []
    for r in records:
        r2 = dict(r)
        r2.pop('extraction_ts', None)  # no duplicar en JSON
        rows.append((run_ts, req_json, json.dumps(r2, ensure_ascii=False, default=_json_default)))

    # bulk insert
    psycopg2.extras.execute_values(
        cur,
        f'INSERT INTO "{schema}".{table} (extraction_ts, request_payload, payload) VALUES %s',
        rows,
        template='(%s, %s::jsonb, %s::jsonb)',
        page_size=1000,
    )

    conn.commit()
    cur.close()
    conn.close()
    print(f' Insertadas {len(rows)} filas en {schema}.{table} (una por lead).')
