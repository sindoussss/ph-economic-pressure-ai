"""When a provider says no, the user has to be told which limit and for how long.

A run died on HTTP 429 with the response headers reading 14,399 of 14,400 daily
requests REMAINING and 1,290 of 6,000 tokens-per-minute left, resetting in 47
seconds. The app reported "Free-tier quota may be exhausted — check your daily
request limit", which named the one limit that was fine. The user saw a stalled
progress bar and a traceback in a console they may not have had open.

Every fact needed to say something true was in the headers.
"""
import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest

from ph_economic_ai.engine import llm

#: The exact headers from the run that produced this test.
LIVE_429_HEADERS = {
    'x-ratelimit-limit-requests': '14400',
    'x-ratelimit-remaining-requests': '14399',
    'x-ratelimit-limit-tokens': '6000',
    'x-ratelimit-remaining-tokens': '1290',
    'x-ratelimit-reset-requests': '6s',
    'x-ratelimit-reset-tokens': '47.1s',
}


@pytest.fixture(autouse=True)
def _no_listeners():
    llm.clear_quota_listeners()
    yield
    llm.clear_quota_listeners()


# ── Reading the headers ───────────────────────────────────────────────────────

@pytest.mark.parametrize('raw,expected', [
    ('47.1s', 47.1),
    ('6s', 6.0),
    ('2m59.56s', 179.56),
    ('1h2m3s', 3723.0),
    ('500ms', 0.5),
    ('30', 30.0),
    (None, None),
    ('garbage', None),
])
def test_reset_durations_parse(raw, expected):
    assert llm._parse_reset(raw) == expected


def test_the_tighter_ceiling_is_the_one_reported():
    """The whole defect in one assertion: tokens were the wall, requests were
    fine, and the app named requests."""
    status = llm._quota_from_headers(LIVE_429_HEADERS, 'groq', 'fast', blocked=True)
    assert status.scope == 'tokens'
    assert status.remaining == 1290
    assert status.reset_seconds == pytest.approx(47.1)


def test_the_message_names_the_limit_and_the_wait():
    status = llm._quota_from_headers(LIVE_429_HEADERS, 'groq', 'fast', blocked=True)
    text = status.describe()
    assert 'per-minute token' in text
    assert '47 seconds' in text
    assert 'daily' not in text


def test_a_daily_exhaustion_reports_hours():
    status = llm.QuotaStatus('groq', 'deep', 'requests', limit=1000, remaining=0,
                             reset_seconds=7200, blocked=True)
    assert 'daily request' in status.describe()
    assert 'hours' in status.describe()


def test_plenty_of_headroom_raises_nothing():
    plenty = dict(LIVE_429_HEADERS, **{'x-ratelimit-remaining-tokens': '5900'})
    assert llm._quota_from_headers(plenty, 'groq', 'fast', blocked=False) is None


def test_a_near_miss_warns_before_the_wall():
    """5% left is enough warning to finish a thought and rare enough not to cry
    wolf on every run."""
    nearly = dict(LIVE_429_HEADERS, **{'x-ratelimit-remaining-tokens': '200'})
    status = llm._quota_from_headers(nearly, 'groq', 'fast', blocked=False)
    assert status is not None and not status.blocked
    assert 'almost' in status.describe()


def test_headerless_providers_are_handled():
    assert llm._quota_from_headers({}, 'gemini', 'fast', blocked=True) is None


# ── Notifying ─────────────────────────────────────────────────────────────────

def test_listeners_receive_the_status():
    seen = []
    llm.on_quota_event(seen.append)
    status = llm._quota_from_headers(LIVE_429_HEADERS, 'groq', 'fast', blocked=True)
    llm._emit_quota(status)
    assert seen and seen[0].scope == 'tokens'


def test_a_broken_listener_cannot_kill_a_run():
    """This fires from a worker thread mid-run. A raising listener must not take
    39 calls' worth of work down with it."""
    good = []

    def explodes(_status):
        raise RuntimeError('listener bug')

    llm.on_quota_event(explodes)
    llm.on_quota_event(good.append)
    llm._emit_quota(llm.QuotaStatus('groq', 'fast', 'tokens', blocked=True))
    assert len(good) == 1


def test_registering_twice_notifies_once():
    seen = []
    llm.on_quota_event(seen.append)
    llm.on_quota_event(seen.append)
    llm._emit_quota(llm.QuotaStatus('groq', 'fast', 'tokens', blocked=True))
    assert len(seen) == 1


# ── The banner ────────────────────────────────────────────────────────────────

def test_the_banner_states_the_limit_the_wait_and_the_consequence(qapp, monkeypatch):
    from ph_economic_ai.ui.quota_notice import QuotaNotice
    monkeypatch.setenv('STRATA_LLM_PROVIDER', 'ollama')
    monkeypatch.setenv('STRATA_LLM_DEEP_PROVIDER', 'groq')

    notice = QuotaNotice()
    assert notice.isHidden()

    status = llm._quota_from_headers(LIVE_429_HEADERS, 'groq', 'deep', blocked=True)
    notice.report(status)

    assert not notice.isHidden()
    assert 'per-minute tokens' in notice._title.text()
    detail = notice._detail.text()
    assert '1,290 of 6,000 left' in detail
    assert '47s' in detail
    # The user must know the run continues on a weaker model.
    assert 'ollama' in detail and 'weaker model' in detail


def test_the_banner_can_be_dismissed(qapp):
    from ph_economic_ai.ui.quota_notice import QuotaNotice
    notice = QuotaNotice()
    notice.report(llm.QuotaStatus('groq', 'fast', 'tokens', limit=6000,
                                  remaining=0, reset_seconds=30, blocked=True))
    assert not notice.isHidden()
    notice.dismiss()
    assert notice.isHidden()


def test_the_countdown_announces_the_refill(qapp):
    from ph_economic_ai.ui.quota_notice import QuotaNotice
    notice = QuotaNotice()
    notice.report(llm.QuotaStatus('groq', 'fast', 'tokens', limit=6000,
                                  remaining=0, reset_seconds=2, blocked=True))
    notice._countdown()
    notice._countdown()
    assert 'refilled' in notice._title.text().lower()


@pytest.fixture(scope='module')
def qapp():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app
