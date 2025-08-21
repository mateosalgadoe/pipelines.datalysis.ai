import os
import time
import random
import email.utils as eut
from datetime import datetime, timezone
from typing import Dict, Any, List

import pandas as pd
import requests

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


def _env(name: str, default: str = None):
    v = os.getenv(name, default)
    return v if v is not None and str(v).strip() != '' else None


def _build_params(base: Dict[str, Any], **overrides) -> Dict[str, Any]:
    out = dict(base)
    for k, v in overrides.items():
        if v is not None and str(v).strip() != '':
            out[k] = v
    return out


def _parse_retry_after(value: str) -> int | None:
    """Devuelve segundos sugeridos por Retry-After (puede ser número o fecha HTTP)."""
    if not value:
        return None
    try:
        return max(0, int(value))
    except Exception:
        try:
            dt = eut.parsedate_to_datetime(value)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            delta = (dt - datetime.now(timezone.utc)).total_seconds()
            return max(0, int(delta))
        except Exception:
            return None


@data_loader
def load_data_from_api(*args, **kwargs) -> pd.DataFrame:
    """
    Descarga histórico paginado de /get-leads con:
      - reintentos y backoff exponencial (+jitter)
      - respeto del header Retry-After si existe (429)
      - pausa entre páginas
    Devuelve un único DataFrame (una fila por lead) con columna extraction_ts.
    """
    base_url = _env('ROCKN_API_BASE_URL')
    token    = _env('ROCKN_API_TOKEN')
    per_page = int(_env('ROCKN_API_PER_PAGE', '100'))
    if not base_url or not token:
        raise RuntimeError('Faltan ROCKN_API_BASE_URL o ROCKN_API_TOKEN en ENV.')

    url = f'{base_url.rstrip("/")}/get-leads'

    # control de ritmo / backoff
    start_page    = int(_env('ROCKN_API_START_PAGE', '1'))
    max_pages     = int(_env('ROCKN_API_MAX_PAGES', '0'))    # 0 = ilimitado
    max_retries   = int(_env('ROCKN_API_MAX_RETRIES', '12'))
    base_sleep    = float(_env('ROCKN_API_RETRY_SLEEP_S', '4'))
    page_sleep    = float(_env('ROCKN_API_PAGE_SLEEP_S', '1.0'))
    backoff_base  = float(_env('ROCKN_API_BACKOFF_BASE_S', '4'))
    backoff_max   = float(_env('ROCKN_API_BACKOFF_MAX_S', '120'))
    jitter_pct    = float(_env('ROCKN_API_JITTER_PCT', '0.3'))
    respect_retry = _env('ROCKN_API_RESPECT_RETRY_AFTER', '1') in ('1', 'true', 'True')

    # filtros (kwargs pisa a ENV)
    def pick(env_name: str, kw_key: str):
        return kwargs.get(kw_key, _env(env_name))

    base_params = {
        'token': token,
        'per_page': per_page,
        'date_from':       pick('ROCKN_DATE_FROM', 'date_from'),
        'date_to':         pick('ROCKN_DATE_TO', 'date_to'),
        'source':          pick('ROCKN_SOURCE', 'source'),
        'state':           pick('ROCKN_STATE', 'state'),
        'state_location':  pick('ROCKN_STATE_LOCATION', 'state_location'),
        'city':            pick('ROCKN_CITY', 'city'),
    }

    session = requests.Session()
    timeout = (5, 45)  # connect, read
    all_rows: List[Dict[str, Any]] = []
    page = start_page
    pages_fetched = 0

    while True:
        if max_pages and pages_fetched >= max_pages:
            break

        params = _build_params(base_params, page=page)

        for attempt in range(1, max_retries + 1):
            try:
                resp = session.get(url, params=params, timeout=timeout)

                # 429 o 5xx -> backoff
                if resp.status_code == 429 or resp.status_code >= 500:
                    # si hay Retry-After, lo respetamos
                    retry_after = _parse_retry_after(resp.headers.get('Retry-After')) if respect_retry else None
                    if retry_after is not None and retry_after > 0:
                        sleep_s = min(retry_after, backoff_max)
                    else:
                        # backoff exponencial con jitter
                        sleep_s = min(backoff_base * (2 ** (attempt - 1)), backoff_max)
                        sleep_s = sleep_s * (1 + random.uniform(0, jitter_pct))
                    time.sleep(sleep_s)
                    if attempt < max_retries:
                        continue
                    resp.raise_for_status()  # agotó reintentos -> lanza

                resp.raise_for_status()
                data = resp.json()
                items = data.get('data', data) if isinstance(data, dict) else data
                if not isinstance(items, list):
                    raise ValueError('La API no devolvió una lista en "data".')

                all_rows.extend(items)
                pages_fetched += 1

                # si esta página trajo menos que per_page, asumimos fin
                if len(items) < per_page:
                    break  # sale del for
                # si no, hay más páginas
                time.sleep(page_sleep)
                break  # página OK -> salimos del for de reintentos

            except Exception:
                if attempt < max_retries:
                    # fallback sleep básico si no era 429/5xx
                    time.sleep(base_sleep)
                    continue
                raise  # agotó reintentos

        # si no hubo items en esta página y es la primera, fin
        if pages_fetched == 0 and len(all_rows) == 0:
            break

        # si esta página fue corta (< per_page) ya no seguimos
        if 'items' in locals() and len(items) < per_page:
            break

        page += 1

    df = pd.DataFrame(all_rows)
    df['extraction_ts'] = pd.Timestamp.utcnow().replace(tzinfo=None)  # UTC naive
    df.columns = [c.strip().replace(' ', '_') for c in df.columns]
    return df


@test
def test_output(df: pd.DataFrame, *args) -> None:
    assert df is not None, 'El loader no devolvió DataFrame.'
    if not df.empty:
        expect = {'id', 'first_name', 'last_name', 'status'}
        assert len(expect & set(df.columns)) > 0, f'Columnas inesperadas: {df.columns}'
