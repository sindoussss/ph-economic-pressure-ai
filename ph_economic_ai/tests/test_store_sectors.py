import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ph_economic_ai.engine.store import AgentTrustStore


def test_update_and_read_sectors(tmp_path):
    s = AgentTrustStore(str(tmp_path / 't.db'))
    rid = s.save_run({'x': 1}, final_estimate=-2.40, confidence_pct=54)
    s.update_run_sectors(rid, 0.50, 0.05)
    run = s.get_recent_runs(1)[0]
    assert run['final_estimate'] == -2.40
    assert run['food_estimate'] == 0.50
    assert run['electricity_estimate'] == 0.05


def test_unset_sectors_are_none(tmp_path):
    s = AgentTrustStore(str(tmp_path / 't2.db'))
    s.save_run({'x': 1}, final_estimate=1.0, confidence_pct=50)
    run = s.get_recent_runs(1)[0]
    assert run['food_estimate'] is None
    assert run['electricity_estimate'] is None


def test_migration_idempotent(tmp_path):
    s = AgentTrustStore(str(tmp_path / 't3.db'))
    s._migrate()   # second call must not raise (columns already present)
    rid = s.save_run({'x': 1}, final_estimate=1.0, confidence_pct=50)
    s.update_run_sectors(rid, 0.1, 0.2)
    assert s.get_recent_runs(1)[0]['food_estimate'] == 0.1
