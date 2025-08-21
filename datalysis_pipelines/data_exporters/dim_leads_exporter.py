import os
import psycopg2
import psycopg2.extras
import pandas as pd
from datetime import datetime

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


def _env(n, d=None):
    v = os.getenv(n, d)
    return v if v is not None and str(v).strip() != '' else None


@data_exporter
def export(df: pd.DataFrame, *args, **kwargs):
    """
    Escribe el DataFrame limpio en clean.dim_leads con UPSERT por lead_id.
    Crea schema y tabla si no existen.
    """
    if df is None or df.empty:
        print('DimLeads: DF vacío, no se exporta.')
        return

    schema = kwargs.get('schema', 'clean')
    table = kwargs.get('table', 'dim_leads')

    conn = psycopg2.connect(
        host=_env("ANALYTICS_POSTGRES_HOST"),
        dbname=_env("ANALYTICS_POSTGRES_DB"),
        user=_env("ANALYTICS_POSTGRES_USER"),
        password=_env("ANALYTICS_POSTGRES_PASSWORD"),
        port=int(_env("ANALYTICS_POSTGRES_PORT", "5432")),
        sslmode=_env("ANALYTICS_POSTGRES_SSLMODE", "require"),
    )
    cur = conn.cursor()

    cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}";')
    cur.execute(f'''
        CREATE TABLE IF NOT EXISTS "{schema}"."{table}" (
            lead_id              BIGINT PRIMARY KEY,
            first_name           TEXT,
            last_name            TEXT,
            agent                TEXT,
            email                TEXT,
            phone                TEXT,
            notes                TEXT,
            source               TEXT,
            city                 TEXT,
            state                TEXT,
            state_location       TEXT,
            address              TEXT,
            status               TEXT,
            disqualified_reason  TEXT,
            created_date         TIMESTAMP NULL,
            modified_date        TIMESTAMP NULL,
            schedule_date        TIMESTAMP NULL,
            extraction_ts        TIMESTAMP NULL
        );
    ''')

    cols = [
        'lead_id','first_name','last_name','agent','email','phone','notes','source',
        'city','state','state_location','address','status','disqualified_reason',
        'created_date','modified_date','schedule_date','extraction_ts'
    ]

    # Convertir pandas NA → None para psycopg2
    def _py(v):
        if pd.isna(v):
            return None
        return int(v) if isinstance(v, (pd.Int64Dtype.type,)) else v

    rows = [tuple(_py(v) for v in r) for r in df[cols].itertuples(index=False, name=None)]

    insert_cols = ','.join(f'"{c}"' for c in cols)
    update_cols = ','.join(f'"{c}"=EXCLUDED."{c}"' for c in cols if c != 'lead_id')

    psycopg2.extras.execute_values(
        cur,
        f'''
        INSERT INTO "{schema}"."{table}" ({insert_cols})
        VALUES %s
        ON CONFLICT ("lead_id") DO UPDATE SET {update_cols};
        ''',
        rows,
        page_size=1000,
    )

    conn.commit()
    cur.close()
    conn.close()
    print(f'DimLeads: upsert {len(rows)} filas en {schema}.{table}.')
