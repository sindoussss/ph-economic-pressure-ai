"""Provenance records must describe the file that is actually committed.

`RSK-007`: results could not be independently rebuilt from original inputs. The
sidecars fix that only if they stay truthful, so the property under test is not
"a record exists" but "the record still matches the bytes".
"""
import json

import pytest

from ph_economic_ai.benchmark import provenance


def _write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(body, encoding='utf-8')
    return p


def test_sidecar_sits_beside_the_file_it_describes(tmp_path):
    csv = _write(tmp_path, 'x.csv', 'date,v\n2020-01,1\n')
    side = provenance.write_record(csv, source='s', params={}, transformations=[])
    assert side.name == 'x.csv.provenance.json'
    assert side.parent == csv.parent


def test_record_captures_source_params_and_checksum(tmp_path):
    csv = _write(tmp_path, 'x.csv', 'date,v\n2020-01,1\n2020-02,2\n')
    provenance.write_record(csv, source='IMF IFS via DBnomics',
                            params={'series': 'M.PH.PCPI_IX'},
                            transformations=['label months YYYY-MM'],
                            units='index', notes='n')
    rec = provenance.load_record(csv)
    assert rec['source'] == 'IMF IFS via DBnomics'
    assert rec['request_params'] == {'series': 'M.PH.PCPI_IX'}
    assert rec['transformations'] == ['label months YYYY-MM']
    assert rec['sha256'] == provenance.sha256_file(csv)
    assert rec['retrieved_at'].endswith('+00:00')


def test_record_reads_the_calendar_back_from_the_written_file(tmp_path):
    csv = _write(tmp_path, 'x.csv', 'date,v\n2020-01,1\n2020-03,3\n')
    provenance.write_record(csv, source='s', params={}, transformations=[])
    rec = provenance.load_record(csv)
    assert rec['rows'] == 2
    assert rec['first_month'] == '2020-01' and rec['last_month'] == '2020-03'
    assert rec['missing_months'] == ['2020-02']


def test_a_changed_file_is_reported_not_silently_accepted(tmp_path):
    """The property that matters. A stale record is worse than no record because it
    looks authoritative."""
    csv = _write(tmp_path, 'x.csv', 'date,v\n2020-01,1\n')
    provenance.write_record(csv, source='s', params={}, transformations=[])
    assert provenance.verify(tmp_path)['ok']

    csv.write_text('date,v\n2020-01,999\n', encoding='utf-8')
    result = provenance.verify(tmp_path)
    assert result['changed'] == ['x.csv']
    assert not result['ok']


def test_a_file_without_a_record_is_reported_but_is_not_a_failure(tmp_path):
    """Not every CSV in the tree is a fetched input, so this is informational."""
    _write(tmp_path, 'x.csv', 'date,v\n2020-01,1\n')
    result = provenance.verify(tmp_path)
    assert result['unrecorded'] == ['x.csv']
    assert result['ok']


def test_an_orphaned_record_is_a_failure(tmp_path):
    csv = _write(tmp_path, 'x.csv', 'date,v\n2020-01,1\n')
    provenance.write_record(csv, source='s', params={}, transformations=[])
    csv.unlink()
    result = provenance.verify(tmp_path)
    assert result['orphaned'] == ['x.csv.provenance.json']
    assert not result['ok']


def test_provenance_never_breaks_a_build_on_an_odd_file(tmp_path):
    """A malformed file must still get a record; provenance is a side effect of
    building and may not take a builder down with it."""
    csv = _write(tmp_path, 'weird.csv', 'not,really\na,csv\n')
    provenance.write_record(csv, source='s', params={}, transformations=[])
    assert provenance.load_record(csv)['sha256'] == provenance.sha256_file(csv)


def test_committed_data_matches_its_records():
    """The shipped invariant: no committed data file may have drifted from the
    provenance record that describes it.

    If this fails, re-run the builder for the named file so its sidecar is
    rewritten. Do not hand-edit the record.
    """
    result = provenance.verify()
    assert result['changed'] == [], (
        f'these committed files no longer match their provenance: {result["changed"]}')
    assert result['orphaned'] == [], (
        f'these records describe files that no longer exist: {result["orphaned"]}')


def test_the_fetched_inputs_are_all_recorded():
    """Every panel the benchmark fetches from a remote source carries a record."""
    result = provenance.verify()
    recorded = set(result['verified'])
    required = {
        'features_monthly.csv', 'features_monthly_long.csv',
        'food_features_monthly.csv', 'electricity_features_monthly.csv',
        'usd_php_monthly.csv', 'ph_cpi_monthly.csv', 'world_bank_ron95.csv',
    }
    assert required <= recorded, f'missing provenance for {sorted(required - recorded)}'


def test_the_cpi_record_names_the_resolved_source():
    """RSK-002 was resolved toward IMF IFS via DBnomics on evidence. The record must
    say so, because the previous label was the whole problem."""
    from ph_economic_ai.benchmark.paths import DATA_DIR
    rec = provenance.load_record(DATA_DIR / 'ph_cpi_monthly.csv')
    assert rec is not None, 'ph_cpi_monthly.csv has no provenance record'
    assert 'IFS' in rec['source'] and 'DBnomics' in rec['source'], rec['source']


# ── The sidecar is written with platform line endings ────────────────────────
#
# Found 2026-08-21, after hitting it by hand while correcting the announced
# series under `RSK-069`. `write_record` used `Path.write_text` with no
# `newline`, so Python's text mode translates '\n' to '\r\n' and every sidecar
# written on Windows lands as CRLF while the same call on Linux CI lands as LF.
#
# `.gitattributes` pins `*.provenance.json` to `eol=lf`, which is why this has
# never broken a build: git normalises the blob on commit. The cost is that the
# file on disk then differs from the checkout it came from, so `git status`
# reports a modification with no content change -- the exact signal `RSK-006`
# spent 61 consecutive red CI runs learning to distrust, and the one the
# artifact-churn note now says to investigate rather than revert.
#
# It is worth fixing in this module specifically because provenance exists to
# make files byte-comparable. A checksum recorder that writes its own output
# with platform-dependent bytes is undermining the property it certifies.

def test_the_sidecar_is_written_with_lf_regardless_of_platform(tmp_path):
    data = tmp_path / 'x.csv'
    data.write_bytes(b'date,value\n2026-01,1\n2026-02,2\n')

    target = provenance.write_record(
        data, source='test', params={}, transformations=['none'])

    raw = target.read_bytes()
    assert b'\r\n' not in raw, 'sidecar written with CRLF; git normalises it, disk does not'
    assert raw.endswith(b'\n')
