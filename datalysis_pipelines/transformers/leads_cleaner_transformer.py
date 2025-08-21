import pandas as pd
import numpy as np
from typing import Any, Dict, List

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


FINAL_COLS = [
    'lead_id',
    'first_name', 'last_name', 'agent', 'email', 'phone', 'notes', 'source',
    'city', 'state', 'state_location', 'address',
    'status', 'disqualified_reason',
    'created_date', 'modified_date', 'schedule_date',
    'extraction_ts',
]


def _to_dataframe(data) -> pd.DataFrame:
    """Admite DataFrame, lista[dict] o dict; devuelve DataFrame."""
    if data is None:
        return pd.DataFrame()
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, list):
        return pd.DataFrame(data)
    if isinstance(data, dict):
        return pd.json_normalize(data)
    try:
        return pd.DataFrame(data)
    except Exception:
        return pd.DataFrame()


def _is_empty(x) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and np.isnan(x):
        return True
    if isinstance(x, str) and x.strip() == '':
        return True
    return False



@transformer
def transform(data, *args, **kwargs) -> pd.DataFrame:   # ← parámetro posicional explícito
    """
    Limpia y tabulariza leads para la dimensión clean.dim_leads.
    """
    df = _to_dataframe(data)                            # ← usa directamente 'data'
    if df.empty:
        return pd.DataFrame(columns=FINAL_COLS)


    # Si viene cargado desde raw con una columna 'payload' (jsonb) → aplanar
    if 'payload' in df.columns:
        payload_df = pd.json_normalize(df['payload'])
        if 'extraction_ts' in df.columns and 'extraction_ts' not in payload_df.columns:
            payload_df['extraction_ts'] = df['extraction_ts'].values
        df = payload_df

    # Eliminar columnas que son listas/dicts, y las 3 explícitas
    drop_exact = {'FollowUp', 'comments', 'emailHistories'}
    for c in list(df.columns):
        if c in drop_exact:
            df.drop(columns=[c], inplace=True, errors='ignore')
        else:
            # si alguna celda es list/dict, fuera
            try:
                if df[c].apply(lambda v: isinstance(v, (list, dict))).any():
                    df.drop(columns=[c], inplace=True)
            except Exception:
                pass

    # Renombres solicitados (acepta ambas variantes de 'disqualified')
    rename_map = {
        'id': 'lead_id',
        'date': 'created_date',
        'disquialified': 'disqualified_reason',
        'disqualified': 'disqualified_reason',
    }
    present = {k: v for k, v in rename_map.items() if k in df.columns}
    if present:
        df = df.rename(columns=present)

    # Garantizar columnas del modelo final
    for c in FINAL_COLS:
        if c not in df.columns:
            df[c] = pd.NA

    # Tipos
    df['lead_id'] = pd.to_numeric(df['lead_id'], errors='coerce').astype('Int64')
    for c in ['created_date', 'modified_date', 'schedule_date', 'extraction_ts']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce', utc=False)

    # Normalizaciones
    def keep_or_unknown(x):
        return 'Unknown' if _is_empty(x) else str(x).strip()

    def lower_or_unknown(x):
        return 'Unknown' if _is_empty(x) else str(x).strip().lower()

    def upper_or_unknown(x):
        return 'Unknown' if _is_empty(x) else str(x).strip().upper()

    # city: solo Unknown para nulos/vacíos
    df['city'] = df['city'].apply(keep_or_unknown)

    # state: MAYÚSCULAS, o Unknown
    df['state'] = df['state'].apply(upper_or_unknown)

    # address: minúsculas, o Unknown
    df['address'] = df['address'].apply(lambda x: 'Unknown' if _is_empty(x) else str(x).strip().lower())

    # status y disqualified_reason: minúsculas (y Unknown si vacío)
    df['status'] = df['status'].apply(lower_or_unknown)
    df['disqualified_reason'] = df['disqualified_reason'].apply(lower_or_unknown)

    # Deduplicación por lead_id usando "lo más nuevo"
    df['_rank_key'] = df[['modified_date', 'created_date', 'extraction_ts']].max(axis=1)
    df = df.sort_values(['lead_id', '_rank_key'], ascending=[True, False]).drop_duplicates('lead_id')
    df = df.drop(columns=['_rank_key'])

    # Orden final
    return df[FINAL_COLS]


@test
def test_output(df: pd.DataFrame, *args) -> None:
    # columnas esperadas
    assert list(df.columns) == FINAL_COLS
    # tipos básicos
    if not df.empty:
        assert 'Unknown' not in (df['state'].dropna().str.contains(r'^\s*$', regex=True)).to_string()
