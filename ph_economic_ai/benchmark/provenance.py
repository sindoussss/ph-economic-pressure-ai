"""Provenance records for every committed data file.

Gate 2 of the publication remediation plan asks for "one provenance record per
input and reproducible raw-to-panel transformation", and `RSK-007` records that
some results cannot be independently rebuilt from original inputs.

The gap was concrete. A reader holding `features_monthly.csv` could not tell which
tickers produced it, which endpoint and parameters were used, when it was
retrieved, or what transformation turned a daily series into a monthly one. That
matters more than usual here: the calendar defect of 2026-07-28 was caused by an
endpoint choice (`interval=1mo`) that appeared nowhere in the repository except as
an argument buried in a function body, so nothing about the committed file
recorded the decision that corrupted it.

Each data file gets a sidecar `<name>.provenance.json` holding the source, the
request parameters, the retrieval timestamp, the transformation chain, the shape
and calendar span, and a SHA-256 of the file itself.

`verify()` is the part with teeth: it recomputes each checksum and reports files
that changed without their provenance being rewritten. A stale provenance record
is worse than none, because it looks authoritative.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

from ph_economic_ai.benchmark.paths import DATA_DIR

SIDECAR_SUFFIX = '.provenance.json'
SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def sidecar_path(output_path: Path) -> Path:
    return Path(output_path).with_suffix(Path(output_path).suffix + SIDECAR_SUFFIX)


def _calendar_span(output_path: Path) -> dict:
    """Row count and month span, read back from the file that was actually written."""
    try:
        import pandas as pd
        df = pd.read_csv(output_path, dtype=str)
        col = 'date' if 'date' in df.columns else df.columns[0]
        months = sorted({str(v)[:7] for v in df[col].dropna()
                         if len(str(v)) >= 7 and str(v)[4] == '-'})
        if not months:
            return {'rows': int(len(df))}
        from ph_economic_ai.benchmark.calendar_index import missing_months
        return {'rows': int(len(df)), 'first_month': months[0], 'last_month': months[-1],
                'missing_months': missing_months(months)}
    except Exception as exc:                      # provenance must never break a build
        return {'error': f'{type(exc).__name__}: {exc}'}


def write_record(output_path, source: str, params: dict, transformations: list,
                 units: str = '', notes: str = '') -> Path:
    """Write the sidecar for a file that has just been built."""
    output_path = Path(output_path)
    record = {
        'schema_version': SCHEMA_VERSION,
        'file': output_path.name,
        'source': source,
        'request_params': params,
        'transformations': transformations,
        'units': units,
        'notes': notes,
        'retrieved_at': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
        'sha256': sha256_file(output_path),
        **_calendar_span(output_path),
    }
    target = sidecar_path(output_path)
    target.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    return target


def load_record(output_path) -> dict | None:
    p = sidecar_path(Path(output_path))
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding='utf-8'))


def verify(data_dir=None) -> dict:
    """Check every data file against its provenance record.

    Returns lists rather than raising, so a caller can report all problems at once.
    `changed` is the one that matters: the file no longer hashes to what its
    provenance claims, so the record describes something else.
    """
    data_dir = Path(data_dir or DATA_DIR)
    verified, changed, unrecorded, orphaned = [], [], [], []

    for path in sorted(data_dir.glob('*.csv')):
        record = load_record(path)
        if record is None:
            unrecorded.append(path.name)
            continue
        if record.get('sha256') != sha256_file(path):
            changed.append(path.name)
        else:
            verified.append(path.name)

    for side in sorted(data_dir.glob(f'*{SIDECAR_SUFFIX}')):
        origin = Path(str(side)[: -len(SIDECAR_SUFFIX)])
        if not origin.exists():
            orphaned.append(side.name)

    return {
        'verified': verified,
        'changed': changed,
        'unrecorded': unrecorded,
        'orphaned': orphaned,
        'ok': not changed and not orphaned,
    }


def main() -> int:
    result = verify()
    print(f'verified   {len(result["verified"])}')
    for name in result['verified']:
        print(f'    ok        {name}')
    for name in result['changed']:
        print(f'    CHANGED   {name}  (file no longer matches its provenance record)')
    for name in result['unrecorded']:
        print(f'    no record {name}')
    for name in result['orphaned']:
        print(f'    orphaned  {name}')
    if result['changed']:
        print('\nA changed file with a stale record is worse than no record. '
              'Re-run the builder so the sidecar is rewritten.')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
