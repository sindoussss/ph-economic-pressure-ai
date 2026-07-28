from __future__ import annotations

import concurrent.futures
import logging
import re
import statistics
import threading
import traceback
from dataclasses import dataclass
from typing import Callable, Optional

import requests
from PyQt6.QtCore import QThread, pyqtSignal

from ph_economic_ai.engine import anchoring, llm, vintage
from ph_economic_ai.engine.rag import RagEngine
from ph_economic_ai.engine.debate import (
    AgentResponse, _MAX_REALISTIC_FUEL_PHP_L, _extract_price, _parse_think)
from ph_economic_ai.engine.live_data import LiveDataBrief


# Fallback price used when live fetch fails (₱/L, NCR unleaded 91 avg).
# Only needs updating if the live fetcher stops working for an extended period.
_FALLBACK_RETAIL_PRICE_PHP: float = 98.82  # NCR Unleaded 91 avg May 20 2026

_PRICE_FETCH_HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}


def fetch_live_retail_price() -> float:
    """Fetch current NCR retail gasoline price from fuelprice.ph (DOE-sourced).

    Parses brand-average prices for all fuel types, takes the median of values
    in the 60–150 range. Falls back to _FALLBACK_RETAIL_PRICE_PHP on any error.

    A caller that needs to know whether the number is REAL must use
    `fetch_live_retail_price_checked`. Silently returning a constant is fine for
    a prompt, where a slightly stale baseline costs nothing, and dangerous for
    grading, where it becomes an observed outcome. Ten stored runs were graded
    against a change of exactly +0.00 because both the run's baseline and the
    grader's "observation" were this constant.
    """
    return fetch_live_retail_price_checked()[0]


def fetch_live_retail_price_checked() -> tuple[float, bool]:
    """The price and whether it was actually fetched.

    `(price, True)` when the scrape succeeded, `(fallback, False)` otherwise.
    Grading must not treat a fallback as an observation: an unavailable price is
    not evidence that the price did not move, and recording it as one produces a
    grade that looks like a quiet week and scores every estimate as wrong by its
    own magnitude.
    """
    try:
        from bs4 import BeautifulSoup
        resp = requests.get(
            'https://www.fuelprice.ph/',
            headers=_PRICE_FETCH_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        for tag in soup(['script', 'style']):
            tag.decompose()
        text = soup.get_text(separator=' ', strip=True)
        # Matches "avg ₱98.82/L" or "avg P98.82/L" — fuelprice.ph brand averages
        hits = [
            float(m) for m in re.findall(
                r'avg\s*[^\d]*(\d{2,3}(?:\.\d{1,2})?)\s*/[Ll]', text
            )
            if 60.0 <= float(m) <= 150.0
        ]
        if hits:
            hits.sort()
            return hits[len(hits) // 2], True   # median across fuel types
    except Exception:
        logging.info('swarm: retail price fetch failed; using the fallback',
                     exc_info=True)
    return _FALLBACK_RETAIL_PRICE_PHP, False

# ── Region list ───────────────────────────────────────────────────────────────
REGIONS: list[str] = [
    'NCR',
    'Central Luzon',
    'Western Visayas',
    'Davao Region',
]

# Pairs: (0,1) and (2,3) → 2 regional judges
REGION_PAIRS: list[tuple[int, int]] = [(0, 1), (2, 3)]

# ── Tier assignments ──────────────────────────────────────────────────────────
# Every bulk agent runs on the fast tier and only the judges get the deep one.
# That looks blunter than the old five-way 3b/7b/14b split, but free-tier
# tokens-per-minute — not requests-per-day — is the binding constraint, and the
# deep tier's TPM ceiling cannot absorb 32 agent calls carrying RAG context.
# Spending it on the 7 judge calls buys more: the judges are what actually
# determine the master verdict.
_ROLE_TIERS: dict[str, str] = {
    'Forecaster':        llm.FAST,
    'DataExtractor':     llm.FAST,
    'Synthesizer':       llm.FAST,
    'Critic':            llm.FAST,
    'ConfidenceScorer':  llm.FAST,
}
_JUDGE_TIER = llm.DEEP

# RegionalJudge.run makes three calls (two defences + a synthesis); MasterJudge
# makes one. Named so expected_call_counts() stays honest if either changes.
_REGIONAL_JUDGE_CALLS = 3
_MASTER_JUDGE_CALLS = 1

# Reserved completion length per call. Module-level so the ablation harness can
# vary it: completions are ~24K of a run's ~44K fast-tier tokens, making this
# the single biggest lever on free-tier run time.
#
# The judge budgets are deliberately generous. A reasoning model on the deep
# tier (deepseek-r1 and friends) spends hundreds of tokens thinking before it
# writes anything, and if the cap lands mid-thought the reply is truncated
# before the ESTIMATE line — which does not error, it just silently yields a
# verdict with no estimate. This is a cap, not a target: models that finish
# early cost nothing extra.
_AGENT_MAX_TOKENS = 750
_JUDGE_MAX_TOKENS = 1800
_MASTER_MAX_TOKENS = 2000
# Single source of truth: the same parse-sanity bound the Forum uses, kept in
# debate.py beside the food and electricity ones so the two debate systems
# cannot drift to different notions of a plausible fuel move.
_MAX_REALISTIC_FUEL_CHANGE = _MAX_REALISTIC_FUEL_PHP_L

# Role processing order within a round (Critic and ConfidenceScorer last so they
# can score agents they've already seen)
_ROLE_ORDER = ['Forecaster', 'DataExtractor', 'Synthesizer', 'Critic', 'ConfidenceScorer']

#: Roles that must read their peers to do their job at all: both score other
#: agents by name, so a blind Critic has nothing to critique. The three
#: estimating roles have no such need, and theirs are the estimates the agreement
#: metric counts — which is why blindness can be applied to exactly those.
_PEER_READING_ROLES = frozenset({'Critic', 'ConfidenceScorer'})


def expected_call_counts() -> dict[str, int]:
    """How many LLM calls one swarm run costs, derived from the swarm's shape.

    Single source of truth for anything that needs the number — the setup
    screen's time estimate, and free-tier quota planning. Derived rather than
    hardcoded because a stale constant is exactly how the old estimate came to
    claim "~371 calls" for a run that actually makes 39.

    This is the clean-path cost. A response with no parseable ESTIMATE line buys
    one extra call to ask for it (`_reask_for_estimate`), capped at one per
    response, so a bad run costs more than this and a good one costs exactly it.
    Counting the retries here would inflate every estimate to plan for a failure
    that usually does not happen.
    """
    alive = len(_ROLE_ORDER)
    per_group = 0
    for _round_num, n_eliminate in _BRACKET:
        per_group += alive
        alive -= n_eliminate

    fast = per_group * len(REGIONS)
    deep = len(REGION_PAIRS) * _REGIONAL_JUDGE_CALLS + _MASTER_JUDGE_CALLS
    return {'fast': fast, 'deep': deep, 'total': fast + deep}


def group_critical_path() -> int:
    """Sequential call-depth of one group, in call durations.

    Round 1 is sequential by design (each agent reads its peers' answers in
    order), so it costs one duration per agent. Later rounds fan out across a
    thread pool and cost roughly one duration each regardless of width.
    """
    if not _BRACKET:
        return 0
    first_round_agents = len(_ROLE_ORDER)
    later_rounds = len(_BRACKET) - 1
    return first_round_agents + later_rounds


def _is_realistic_fuel_change(value: Optional[float]) -> bool:
    return value is not None and abs(value) <= _MAX_REALISTIC_FUEL_CHANGE


#: The direction line agents and judges must state before their number.
#:
#: The problem it solves is not disagreement, it is EMISSION. Live statements
#: reasoned correctly and then signed the opposite way:
#:
#:     "consumer sees a reduction in retail gasoline prices"
#:     ESTIMATE: +PHP4.50/L
#:
#: A whole region did it at once — Western Visayas and Davao returned +1.00 at
#: 97 percent internal agreement against a -3.28 anchor, and confident wrongness
#: reads as consensus. Committing to a word first is a cheap scaffold that makes
#: the sign explicit, and it gives the parser something to check the number
#: against instead of trusting a lone character.
DIRECTION_LINE = 'DIRECTION: UP or DIRECTION: DOWN or DIRECTION: FLAT'

_DIRECTION_RE = re.compile(
    r'DIRECTION\s*:?\s*\**\s*(UP|DOWN|FLAT|RISE|RISING|FALL|FALLING|'
    r'INCREASE|DECREASE|UNCHANGED|STEADY)\b',
    re.IGNORECASE)
_DOWN_WORDS = {'down', 'fall', 'falling', 'decrease'}
_FLAT_WORDS = {'flat', 'unchanged', 'steady'}


def parse_direction(text: str) -> Optional[int]:
    """-1, 0, +1 from the agent's DIRECTION line, or None if it gave none.

    The LAST such line wins, matching `parse_fuel_estimate`: a model that
    restates its answer means the final one.
    """
    hits = _DIRECTION_RE.findall(text or '')
    if not hits:
        return None
    word = hits[-1].lower()
    if word in _FLAT_WORDS:
        return 0
    return -1 if word in _DOWN_WORDS else 1


def direction_contradicts(direction: Optional[int],
                          estimate: Optional[float]) -> bool:
    """True when a stated direction and a signed estimate disagree.

    FLAT is deliberately permissive: an agent that says flat and writes -0.05 is
    being consistent, not contradictory, so only a sign flip counts. Absence of
    either is not a contradiction — there is nothing to contradict.
    """
    if direction is None or estimate is None or direction == 0:
        return False
    return (estimate > 0) != (direction > 0)


def parse_fuel_estimate(text: str) -> tuple[Optional[float], Optional[float]]:
    """Parse a PHP/L fuel change into (accepted, rejected).

    Returns the value in the first slot if it survives the plausibility guard,
    otherwise in the second. Separating the two lets callers tell "the judge
    never gave a number" apart from "the judge gave one and we threw it away" —
    a distinction the report previously collapsed into a bare em dash.
    """
    estimate = None
    # Tolerance-band echoes ("stay within +/-P1.00/L") contain peso amounts that
    # are instructions, not answers — stripped before any matching.
    from ph_economic_ai.engine.debate import _TOLERANCE_BAND_RE
    text = _TOLERANCE_BAND_RE.sub(' ', text)
    # The currency symbol may come BEFORE the sign. A model writing the peso
    # naturally produces "ESTIMATE: PHP-0.50/L", which the sign-first pattern
    # missed entirely, and a missed estimate is not a small loss: it makes
    # compute_combined_score return 0.0, so an entire group ties at 0.00 and the
    # elimination bracket stops measuring anything. Word forms are accepted for
    # the same reason.
    estimate_lines = re.findall(
        r'ESTIMATE\s*:\s*\**\s*(?:₱|PHP|P|â‚±)?\s*'
        r'([+\-]|minus|plus)?\s*(?:₱|PHP|P|â‚±)?\s*'
        r'(\d+(?:\.\d+)?)\s*/?\s*(?:PHP)?\s*L?',
        text,
        flags=re.IGNORECASE,
    )
    if estimate_lines:
        sign, raw = estimate_lines[-1]
        if not sign.strip():
            # Unsigned defaults to positive, EXCEPT when the agent's own words
            # argue a fall — then the parse contradicts the statement and is
            # refused rather than guessed. This is the +1.0-among-all-negatives
            # artifact from a live run: one stamped-positive unsigned number
            # became the outlier that dragged the centre and the agreement.
            from ph_economic_ai.engine.debate import _DECREASE_CUES, _INCREASE_CUES
            if _DECREASE_CUES.search(text) and not _INCREASE_CUES.search(text):
                return None, None
        negative = sign.strip().lower() in ('-', 'minus')
        estimate = (-1 if negative else 1) * float(raw)
    else:
        estimate = _extract_price(text)

    if estimate is None:
        return None, None
    if not _is_realistic_fuel_change(estimate):
        logging.info(
            'swarm: rejected fuel estimate %+.2f PHP/L (outside +/-%.0f)',
            estimate, _MAX_REALISTIC_FUEL_CHANGE,
        )
        return None, estimate
    return estimate, None


def _extract_fuel_change(text: str) -> Optional[float]:
    """Extract a signed PHP/L fuel price change, rejecting absolute-price parses."""
    accepted, _ = parse_fuel_estimate(text)
    return accepted


_ESTIMATE_RETRY_PROMPT = (
    'Your reply did not end with a usable estimate line. Do not repeat your '
    'analysis. Reply with that one line and nothing else, using the sign that '
    'matches the reasoning you just gave:\n'
    'ESTIMATE: +₱X.XX/L   or   ESTIMATE: -₱X.XX/L'
)
# Enough room for a reasoning model to think briefly and emit one line. A cap
# that lands mid-thought yields nothing, which is the failure being fixed.
_AGENT_RETRY_MAX_TOKENS = 250
_JUDGE_RETRY_MAX_TOKENS = 800


_DIRECTION_RETRY_PROMPT = (
    'Your DIRECTION line and the sign of your ESTIMATE disagree with each other. '
    'You said prices move {stated}, then gave {estimate:+.2f} PHP/L. Only one of '
    'those can be what you meant.\n'
    'Do not repeat your analysis. Reply with exactly these two lines, consistent '
    'with each other:\n'
    'DIRECTION: UP or DIRECTION: DOWN or DIRECTION: FLAT\n'
    'ESTIMATE: +PHP X.XX/L or ESTIMATE: -PHP X.XX/L'
)


def _resolve_direction_conflict(
    messages: list[dict], statement: str, direction: Optional[int],
    estimate: Optional[float], *, tier: str, max_tokens: int, seed: int,
) -> tuple[str, Optional[float]]:
    """Ask an agent which it meant when its direction and sign disagree.

    Returns the statement and the estimate to keep. On an unresolved conflict the
    estimate is REFUSED rather than guessed, which matches the existing rule for
    an unsigned number contradicted by its own prose: the agent has made one
    claim too many and which it meant is not knowable from outside.

    Refusing costs a data point. Keeping a contradicted number costs the metric
    its meaning, because a sign flip is not a small error here — it is the
    difference between prices rising and falling, and a region can agree on it at
    97 percent while being inverted.
    """
    if not direction_contradicts(direction, estimate):
        return statement, estimate

    stated = 'DOWN' if direction < 0 else 'UP'
    logging.info('swarm: direction %s contradicts estimate %+.2f, re-asking',
                 stated, estimate)
    followup = list(messages) + [
        {'role': 'assistant', 'content': statement},
        {'role': 'user', 'content': _DIRECTION_RETRY_PROMPT.format(
            stated=stated, estimate=estimate)},
    ]
    try:
        full = ''.join(llm.stream(followup, tier=tier, max_tokens=max_tokens,
                                  seed=seed))
    except Exception:
        logging.exception('swarm: direction retry call failed')
        return statement, None
    _, reply = _parse_think(full)
    new_estimate = _extract_fuel_change(reply)
    new_direction = parse_direction(reply)
    if new_estimate is None or direction_contradicts(new_direction, new_estimate):
        logging.info('swarm: direction conflict unresolved, refusing the estimate')
        return statement, None
    return f'{statement.rstrip()}\n{reply.strip()}', new_estimate


def _reask_for_estimate(
    messages: list[dict], statement: str, *, tier: str, max_tokens: int, seed: int,
) -> tuple[str, Optional[float]]:
    """Ask once more for a missing ESTIMATE line. Returns (statement, estimate).

    A response with no parseable estimate is not a quiet loss. It scores 0.0 in
    `compute_combined_score`, drops out of the agreement population, and if it
    belongs to the group's survivor it takes that whole region out of the
    headline. Run 28 lost 5 of 18 responses this way, 28 percent, and the missing
    NCR number is what dropped a region pair from the master verdict.

    The original statement is kept as the agent's reasoning, because that is what
    the Critic reads and the report shows, with the follow-up line appended. Only
    a failure costs a call, and only ever one.
    """
    followup = list(messages) + [
        {'role': 'assistant', 'content': statement},
        {'role': 'user', 'content': _ESTIMATE_RETRY_PROMPT},
    ]
    try:
        full = ''.join(llm.stream(followup, tier=tier, max_tokens=max_tokens,
                                  seed=seed))
    except Exception:
        # A failed retry must not take the original answer down with it.
        logging.exception('swarm: estimate retry call failed')
        return statement, None
    _, reply = _parse_think(full)
    estimate = _extract_fuel_change(reply)
    if estimate is None:
        logging.info('swarm: estimate retry produced no usable number either')
        return statement, None
    return f'{statement.rstrip()}\n{reply.strip()}', estimate


def _vintage_seed(*parts: object) -> int:
    """A per-call seed keyed on the run's VINTAGE plus a local identifier.

    The vintage — this day, this DOE pricing week — is what holds still while the
    market does. The extra parts keep the agents from collapsing into one voice.

    This used to hash the scenario dict, and that is why ADR-002's reproducibility
    claim never held. `oil_pct`, `usd_pct` and `demand_index` are recomputed from
    live Yahoo and Open-Meteo values on every run and `current_price` is
    rescraped, so a 0.01 percent move in Brent handed every agent in the swarm a
    different seed. Of the eight runs on 2026-07-27, five had a distinct scenario
    and therefore a distinct seed for every call in the run.

    Quantising the scenario onto a grid was tried and rejected: two runs either
    side of a grid line still diverge, and runs 27 and 28 sat 0.07 percentage
    points apart across one. Keying on the window instead has no boundary to
    straddle, and it is what `engine/forum.py` has always done — `derive_seed(
    as_of, window, sector, agent, round)` — which is why the Forum is the one
    part of the app whose seeds were already same-day stable.

    The scenario does not need to be in the seed. It is already in the PROMPT, so
    moving a slider still changes the answer; what it no longer does is reshuffle
    the sampler underneath an unchanged question.
    """
    v = vintage.vintage()
    return llm.derive_seed(v['fuel_cycle'], v['day'], *parts)


#: How close two estimates must sit to count as agreeing, in ₱/L.
#:
#: 0.50, matching `forum._BAND['gas']`. It was 1.00, which the project's own
#: control test had already disqualified — the study is recorded in the vault and
#: the Forum was moved to 0.50 on the strength of it, while the swarm was left
#: behind. Re-run here against the same controls:
#:
#:     two clusters 1.5 apart   band 1.00 -> 62%    band 0.50 -> 50%
#:     two clusters 0.9 apart   band 1.00 -> 92%    band 0.50 -> 55%
#:
#: A room split into two halves 0.9 apart scoring 92 percent is the metric
#: reporting a split as a consensus. 0.50 is not chosen because it flatters the
#: number — it lowers every figure the app has ever shown — but because it is the
#: widest band at which both controls hold.
#:
#: Note what it does NOT fix: a room collapsed onto two nearly identical values
#: still scores 99 either way, because those agents genuinely are close. That
#: case is caught by `agreement_distinct` and `opening_diversity` instead. The
#: two guards are complementary, which is why both exist.
_AGREEMENT_BAND = 0.50
#: The wider "nearly agreeing" ring, kept at 1.5x the main band as before.
_AGREEMENT_NEAR_BAND = 0.75


def _robust_confidence_pct(estimates: list[float], final_estimate: Optional[float]) -> int:
    """How tightly a set of estimates agrees with itself, robust to outliers.

    The old calculation used standard deviation across all intermediate values,
    so one bad parse like -92.30 could force a 10-15% confidence floor. This
    version discards impossible fuel changes and scores how tightly the usable
    estimates cluster around their own medoid — `final_estimate` is accepted for
    call-site compatibility and deliberately unused, see the centring note below.

    Prefer `measure_agreement`, which returns the population size alongside the
    percentage. A bare percentage cannot tell a reader whether it came from
    twenty agents or from two.
    """
    valid = [e for e in estimates if _is_realistic_fuel_change(e)]
    if len(valid) < 2:
        # Not measurable — and specifically NOT 65 percent, which is what this
        # returned for a single estimate. A lone number agrees with itself by
        # construction, so scoring it was a hardcoded confidence dressed as a
        # measurement: the same defect as the 0.75 the regional cards used to
        # ship. 0 is safe to reserve for "nothing to measure" because a real
        # room can never reach it — the centre always agrees with itself, so the
        # floor for n estimates is 1/n.
        return 0
    # Centre on the AGENTS, not on the judge's verdict.
    #
    # This used to centre on `final_estimate`. That number is anchored: when the
    # agents produce nothing usable, or the magnitude guard clamps them, it is
    # pulled onto the mechanical pass-through. Agreement then measured "how close
    # are the agents to the anchor", not "do the agents agree with each other",
    # while the card labelled it agent agreement. Measured on a live run with the
    # agents clustered near -2.0 and the judge on the anchor at -1.03, the same
    # estimates scored 65 percent against the judge and 93 percent against
    # themselves: a 28-point artefact.
    #
    # Medoid rather than median, matching the Forum: models quote on a coarse
    # grid, so an abstract median can land in a gap between clusters where the
    # band reaches nobody. The centre of a consensus measure has to be a value an
    # agent actually gave.
    center = min(valid, key=lambda c: (sum(abs(e - c) for e in valid), c))

    close = [e for e in valid if abs(e - center) <= _AGREEMENT_BAND]
    near = [e for e in valid if abs(e - center) <= _AGREEMENT_NEAR_BAND]
    # Spread over the WHOLE room, not just the agreeing subset. Measuring it on
    # `close` only asked "how tight are the agents who already agree", which is
    # near-tautological: a room split 50/50 into two tight clusters scored 61
    # percent because the half inside the band was compact. Over all estimates
    # the same split reads 38 percent, which is what a split should read.
    spread = statistics.pstdev(valid) if len(valid) > 1 else 0.0

    agreement_score = len(close) / len(valid)
    near_score = len(near) / len(valid)
    spread_score = max(0.0, 1.0 - min(spread / 1.50, 1.0))
    confidence = 0.50 * agreement_score + 0.25 * near_score + 0.25 * spread_score
    # No artificial floor or ceiling. A 10 percent floor overstates a room that
    # genuinely agrees on nothing, and a 95 percent cap understates one that is
    # unanimous -- the Forum reports both honestly and these two halves of the app
    # should not disagree about what 100 percent means.
    return int(round(confidence * 100))


#: How close an estimate must sit to the anchor to count as repeating it rather
#: than arriving at it. Two decimals is the resolution the models quote at, so
#: equality there is repetition rather than coincidence.
_ANCHOR_ECHO_TOLERANCE = 0.005

#: Fewest estimates that can carry an agreement figure. Two is a coin, not a
#: measurement: on a two-point sample this metric is effectively binary — a gap
#: of ₱1.00 scores 92%, ₱1.01 scores 67%, and anything past ₱3.00 pins to its
#: 38% floor no matter how far apart the pair actually is. The master verdict
#: was scoring exactly two survivors, which is how a room reading 74% and 53%
#: on its regional cards published 38% as its headline.
_AGREEMENT_QUORUM = 2


def measure_agreement(estimates: list[Optional[float]]) -> tuple[int, int]:
    """Agreement percentage and the number of estimates it was measured over.

    The count is not decoration. "38% agreement" means one thing over twenty
    agents and something else entirely over two, and the report has no way to
    tell those apart from the percentage alone — so the population travels with
    the number everywhere it goes.
    """
    valid = [e for e in estimates if _is_realistic_fuel_change(e)]
    return _robust_confidence_pct(valid, None), len(valid)


def agent_responses_of(history: list) -> list:
    """The responses in one group's history that carry a usable estimate.

    Paired with `agent_estimates_of` so the diversity measures and the agreement
    percentage describe the same population rather than two different ones.
    """
    return [r for r in history
            if _is_realistic_fuel_change(getattr(r, 'price_estimate', None))]


def opening_diversity(responses: list) -> float:
    """Share of agents whose OPENING READ is distinct, 0 to 1.

    The signal the percentage cannot carry. Two agents independently reaching
    -2.10 and sixteen agents copying -2.10 produce an identical estimate vector,
    so no function of the numbers alone separates consensus from repetition. The
    prose does: a copied answer arrives in copied words.

    One statement per agent, its earliest. The first version divided distinct
    openings by RESPONSES, and an agent speaks in both rounds: 32 responses came
    from 20 agents, so the ceiling was 20/32 = 0.625 rather than 1.0 and a
    threshold of 0.5 sat at four fifths of an unreachable maximum. It would have
    fired on almost every healthy run.

    The earliest round is also the right one to read. That is the blind round,
    where an agent forms its own view, and it is exactly where peer contamination
    would show. A round-2 statement is SUPPOSED to respond to the room.
    """
    first_by_agent: dict = {}
    for r in responses:
        name = getattr(r, 'agent_name', None)
        rnd = getattr(r, 'round_num', 0) or 0
        if name is None:
            continue
        prior = first_by_agent.get(name)
        if prior is None or rnd < (getattr(prior, 'round_num', 0) or 0):
            first_by_agent[name] = r

    openings = [' '.join((getattr(r, 'statement', '') or '').split())[:80]
                for r in first_by_agent.values()]
    openings = [o for o in openings if o]
    if not openings:
        return 0.0
    return round(len(set(openings)) / len(openings), 3)


def agent_estimates_of(history: list) -> list[float]:
    """Every usable price estimate in one group's debate history.

    All rounds, deliberately. The regional cards measure agreement over exactly
    this population, so the master verdict has to as well or the same words on
    the same screen describe two different measurements.
    """
    return [r.price_estimate for r in history
            if _is_realistic_fuel_change(getattr(r, 'price_estimate', None))]


# ── Physics anchor, shared by everyone who estimates ──────────────────────────
# This block used to exist only inside MasterJudge. Twenty agents and two
# regional judges produced their numbers without ever being told the mechanical
# pass-through, and the single call that did see it then clamped the result back
# to physics at the very end.
#
# That is the wrong place for a prior. Withholding the one input that sets the
# scale and then measuring how far the agents landed from each other measures
# the absence of information, not disagreement about the economy — and it is not
# a small effect: in run 28 the whole Davao group came in at +₱2.00/L against a
# −₱1.02/L pass-through, the wrong SIGN, and the room spanned ₱7.00. The anchor
# is accounting, not opinion (see engine/anchoring.py), so there is no analytical
# independence to protect by keeping it from them.
#
# Diversity is preserved the same way it is with the RAG corpus: by giving every
# agent the same facts and letting the role prompt decide what they do with them.
# The block below is a baseline to reason from and to depart from with a reason,
# not a target to restate.

def compute_physical_anchor(scenario: dict,
                            data_brief: Optional['LiveDataBrief'] = None) -> float:
    """The mechanical oil→pump pass-through for this scenario, in ₱/L.

    Live Brent/FX when the brief has them, reference values otherwise — the
    anchor is a scale, not a precise forecast, so stale inputs cost cents.
    """
    kwargs = {}
    if data_brief is not None:
        if getattr(data_brief, 'brent', None):
            kwargs['brent_usd'] = data_brief.brent
        if getattr(data_brief, 'usd_php', None):
            kwargs['fx_php_per_usd'] = data_brief.usd_php
    return anchoring.fuel_passthrough_anchor(
        scenario.get('oil_pct', 0.0), scenario.get('usd_pct', 0.0), **kwargs
    )


def anchor_prompt_block(anchor: float) -> str:
    """The pass-through, worded so a weak model treats it as a floor to reason
    from rather than a number to echo."""
    return (
        f"MECHANICAL PASS-THROUGH: the DOE auto-pricing formula implies a pump "
        f"change of {anchor:+.2f} ₱/L from this oil and FX move alone (crude cost "
        f"per litre revalued at the exchange rate, plus 12% VAT). This is the "
        f"physical baseline, not a target. Start from it and adjust only for what "
        f"it cannot see — a fuel subsidy release, an excise suspension, a refinery "
        f"outage, your region's freight premium, competitive lag. If your own "
        f"number lands more than ₱2.00/L from it, name the factor that accounts "
        f"for the gap."
    )


# ── Data structures ───────────────────────────────────────────────────────────
@dataclass
class SwarmAgent:
    name: str
    role: str
    tier: str                  # llm.FAST | llm.DEEP — resolved to a model at call time
    group_id: int
    region_name: str
    system_prompt: str
    rag_sources: list[str]
    is_alive: bool = True
    combined_score: float = 0.0


@dataclass
class GroupSurvivor:
    group_id: int
    region_name: str
    response: AgentResponse
    combined_score: float
    agent_role: str
    agent_model: str


@dataclass
class RegionalVerdict:
    judge_id: int
    region_pair: tuple[str, str]
    estimate: Optional[float]
    confidence: float
    reasoning: str
    survivor_names: tuple[str, str]
    # Set when the judge did produce a number but it failed the plausibility
    # guard. Lets the report explain a missing estimate instead of showing a
    # bare dash that reads as a crash.
    rejected_estimate: Optional[float] = None


@dataclass
class MasterVerdict:
    final_estimate: Optional[float]
    confidence_pct: int
    dissenting_regions: list[str]
    reasoning: str
    regional_verdicts: list[RegionalVerdict]
    # How many agent estimates `confidence_pct` was measured over, and how many
    # of the region groups met quorum out of how many ran. Both are reported on
    # the card: a headline reading 63% over 12 agents in 3 of 4 regions is a
    # claim a reader can check, and one reading 38% over 2 is a claim they can
    # reject. 0 estimates means not measurable, which is not the same as 0%.
    agreement_n: int = 0
    agreement_regions: tuple[int, int] = (0, 0)
    # How many of those estimates are the physical anchor repeated verbatim.
    #
    # Every agent is given the mechanical pass-through (ADR-004), which is what
    # stopped a room spanning ₱7 with a whole region on the wrong sign. The cost
    # is that an agent restating the anchor is scored as an agent independently
    # arriving at it, and the metric cannot tell those apart. Measured across two
    # live runs: 25.0 percent of local estimates and 19.4 percent of hosted ones
    # were the anchor to two decimals, and on the hosted run it was the single
    # most common value.
    #
    # So the percentage alone overstates independent corroboration by about a
    # quarter. Reporting the echo share beside it is what keeps the number
    # readable: "52%, of which a fifth is agents restating the anchor" is a claim
    # a reader can weigh; "52%" is not.
    agreement_echo_n: int = 0
    # How many DISTINCT values those estimates take, and how varied the
    # statements behind them were.
    #
    # The percentage saturates long before the room is actually in agreement. A
    # blind-arm experiment measured the shipped swarm at 100 percent agreement
    # over 32 estimates taking exactly TWO distinct values across a ₱0.26 span.
    # Blinding the estimating roles tripled the distinct values and widened the
    # spread nine-fold, and the percentage moved 8 points.
    #
    # So the number cannot stand alone. `agreement_distinct` says how many
    # different answers the room actually produced, and `agreement_diversity`
    # says whether the statements behind them were written independently. Both
    # are reported beside the percentage, because 100 percent over 2 values is a
    # collapsed room and 92 percent over 6 is a working one.
    agreement_distinct: int = 0
    agreement_diversity: float = 0.0
    # The retail price this run actually reasoned from. The orchestrator scrapes
    # it into its OWN copy of the scenario, so without carrying it back the caller
    # stores a scenario the run never saw — and grading compares the observed
    # price against that stored baseline.
    #
    # It went unnoticed because the stale value was plausible: every graded run
    # recorded an "actual change" of exactly -14.44 PHP/L, which is 84.38 minus
    # the 98.82 fallback constant rather than any week's move. That fed
    # `compute_accuracy_score`, which floors at a 3.00 error, so every agent
    # scored 0.0 on accuracy and trust collapsed toward 0.4 x internal. Seven of
    # twenty agents were benched by a number that never described a real outcome.
    current_price: Optional[float] = None
    regional_estimates: Optional[dict] = None  # {region_name: Optional[float]}
    all_responses: list = None   # list[AgentResponse] from all group arenas
    # Physics-anchored reconciliation (engine/anchoring.py): the mechanical
    # pass-through the estimate was checked against, and how the final number
    # was arrived at — 'agent' (model agreed with physics), 'clamped' (model
    # drifted, pulled back), or 'anchor' (model unusable, physics stood in).
    physical_anchor: Optional[float] = None
    estimate_source: str = 'agent'

    def __post_init__(self):
        if self.all_responses is None:
            self.all_responses = []


# ── RAG source assignments per role ───────────────────────────────────────────
# Every role READS every source. The role still SPEAKS to its own lane, which is
# where the diversity of a swarm actually comes from.
#
# Splitting the corpus by role made agents disagree because they were shown
# different facts, which is information asymmetry rather than analysis. The
# ConfidenceScorer saw 2 of 6 sources while scoring everyone else's confidence,
# and the Forecaster never saw BusinessWorld or ManilaBulletin at all. Agents
# cannot converge on evidence they were never given, and a disagreement caused by
# a missing document is not a signal about the economy.
#
# Diversity is preserved by RESPONSIBILITY (the role prompt), not by IGNORANCE
# (a withheld document). If this ever collapses the roster into one voice, the
# fix is a stronger role prompt, not a narrower corpus -- statement diversity is
# measured alongside agreement whenever this is changed.
_ALL_FUEL_SOURCES = ['DOEBulletin', 'PHRetailFuel', 'YahooFinanceCrude',
                     'YahooFinanceForex', 'ManilaBulletin', 'BusinessWorld',
                     'neda_2024_2026']
_ROLE_RAG: dict[str, list[str]] = {
    'Forecaster':       list(_ALL_FUEL_SOURCES),
    'DataExtractor':    list(_ALL_FUEL_SOURCES),
    'Synthesizer':      list(_ALL_FUEL_SOURCES),
    'Critic':           list(_ALL_FUEL_SOURCES),
    'ConfidenceScorer': list(_ALL_FUEL_SOURCES),
}


# ── All 17 PH administrative regions ─────────────────────────────────────────
# anchor: index into swarm group survivors (0=NCR, 1=Central Luzon, 2=W.Visayas, 3=Davao)
# multiplier: freight/logistics premium over NCR (applied to price change magnitude)
ALL_REGIONS: list[dict] = [
    # Luzon
    {'name': 'NCR',               'code': 'NCR',   'multiplier': 1.00, 'anchor': 0,
     'nx': 0.64, 'ny': 0.33, 'isle': 'L'},
    {'name': 'Ilocos Region',     'code': 'I',     'multiplier': 1.05, 'anchor': 0,
     'nx': 0.36, 'ny': 0.14, 'isle': 'L'},
    {'name': 'Cagayan Valley',    'code': 'II',    'multiplier': 1.06, 'anchor': 0,
     'nx': 0.72, 'ny': 0.09, 'isle': 'L'},
    {'name': 'Central Luzon',     'code': 'III',   'multiplier': 1.02, 'anchor': 1,
     'nx': 0.60, 'ny': 0.25, 'isle': 'L'},
    {'name': 'CALABARZON',        'code': 'IVA',   'multiplier': 1.03, 'anchor': 1,
     'nx': 0.70, 'ny': 0.40, 'isle': 'L'},
    {'name': 'MIMAROPA',          'code': 'IVB',   'multiplier': 1.08, 'anchor': 1,
     'nx': 0.44, 'ny': 0.44, 'isle': 'L'},
    {'name': 'Bicol Region',      'code': 'V',     'multiplier': 1.06, 'anchor': 1,
     'nx': 0.82, 'ny': 0.44, 'isle': 'L'},
    {'name': 'CAR',               'code': 'CAR',   'multiplier': 1.08, 'anchor': 0,
     'nx': 0.60, 'ny': 0.15, 'isle': 'L'},
    # Visayas
    {'name': 'Western Visayas',   'code': 'VI',    'multiplier': 1.05, 'anchor': 2,
     'nx': 0.36, 'ny': 0.56, 'isle': 'V'},
    {'name': 'Central Visayas',   'code': 'VII',   'multiplier': 1.04, 'anchor': 2,
     'nx': 0.62, 'ny': 0.57, 'isle': 'V'},
    {'name': 'Eastern Visayas',   'code': 'VIII',  'multiplier': 1.07, 'anchor': 2,
     'nx': 0.82, 'ny': 0.53, 'isle': 'V'},
    # Mindanao
    {'name': 'Zamboanga',         'code': 'IX',    'multiplier': 1.08, 'anchor': 3,
     'nx': 0.26, 'ny': 0.70, 'isle': 'M'},
    {'name': 'Northern Mindanao', 'code': 'X',     'multiplier': 1.06, 'anchor': 3,
     'nx': 0.56, 'ny': 0.70, 'isle': 'M'},
    {'name': 'Caraga',            'code': 'XIII',  'multiplier': 1.07, 'anchor': 3,
     'nx': 0.82, 'ny': 0.74, 'isle': 'M'},
    {'name': 'Davao Region',      'code': 'XI',    'multiplier': 1.05, 'anchor': 3,
     'nx': 0.72, 'ny': 0.82, 'isle': 'M'},
    {'name': 'SOCCSKSARGEN',      'code': 'XII',   'multiplier': 1.07, 'anchor': 3,
     'nx': 0.56, 'ny': 0.87, 'isle': 'M'},
    {'name': 'BARMM',             'code': 'BARMM', 'multiplier': 1.10, 'anchor': 3,
     'nx': 0.38, 'ny': 0.84, 'isle': 'M'},
]


def derive_regional_estimates(
    base_estimate: Optional[float],
    anchor_estimates: Optional[dict] = None,
) -> dict:
    """Derive per-region price change estimates for all 17 PH regions.

    anchor_estimates maps group_id (0-3) → survivor price estimate.
    Falls back to base_estimate when an anchor is missing.
    Multiplies the anchor change by the region's logistics freight factor.
    """
    anchors = anchor_estimates or {}
    result: dict[str, Optional[float]] = {}
    for reg in ALL_REGIONS:
        anchor_est = anchors.get(reg['anchor'], base_estimate)
        if anchor_est is None:
            anchor_est = base_estimate
        result[reg['name']] = (
            round(anchor_est * reg['multiplier'], 2) if anchor_est is not None else None
        )
    return result


def _make_system_prompt(role: str, region: str, current_price: float = _FALLBACK_RETAIL_PRICE_PHP) -> str:
    price_anchor = (
        f"IMPORTANT: The current DOE-published retail gasoline price in the Philippines "
        f"is approximately ₱{current_price:.2f}/L (unleaded 95). "
        f"Your ESTIMATE must be a realistic price CHANGE from this baseline — "
        f"typical weekly adjustments are ±₱0.20 to ±₱3.00/L. "
        f"Do NOT output the absolute price; output only the signed change. "
    )
    base = (
        f"You are analyzing fuel price dynamics specifically for the {region} region "
        f"of the Philippines. {price_anchor}"
    )
    if role == 'Forecaster':
        return (
            base +
            "Project the short-term retail gasoline price CHANGE for this region "
            "based on crude oil prices, forex, and regional demand patterns. "
            "End with BOTH lines, and make them agree:\n"
            "DIRECTION: UP or DIRECTION: DOWN or DIRECTION: FLAT\n"
            "ESTIMATE: +₱X.XX/L or ESTIMATE: -₱X.XX/L"
        )
    if role == 'DataExtractor':
        return (
            base +
            "Extract and highlight the most relevant economic data points for this region "
            "(infrastructure, income levels, freight costs, demand patterns). "
            "Reference specific peso values from the DOE bulletin context where available. "
            "End with BOTH lines, and make them agree:\n"
            "DIRECTION: UP or DIRECTION: DOWN or DIRECTION: FLAT\n"
            "ESTIMATE: +₱X.XX/L or ESTIMATE: -₱X.XX/L"
        )
    if role == 'Synthesizer':
        return (
            base +
            "Integrate all data and prior estimates into a coherent regional price view. "
            "Resolve contradictions between other agents' estimates. "
            "End with BOTH lines, and make them agree:\n"
            "DIRECTION: UP or DIRECTION: DOWN or DIRECTION: FLAT\n"
            "ESTIMATE: +₱X.XX/L or ESTIMATE: -₱X.XX/L"
        )
    if role == 'Critic':
        return (
            base +
            "Challenge the reasoning of other agents in your group. Identify flaws, "
            "unsupported claims, and biases. Give your own estimate, then rate each "
            "agent's reasoning quality using this exact format on separate lines: "
            "SCORE: <agent_name>: X  (1–10, no /10 suffix). "
            "End with BOTH lines, and make them agree:\n"
            "DIRECTION: UP or DIRECTION: DOWN or DIRECTION: FLAT\n"
            "ESTIMATE: +₱X.XX/L or ESTIMATE: -₱X.XX/L"
        )
    if role == 'ConfidenceScorer':
        return (
            base +
            "Evaluate confidence in each agent's price estimate based on evidence "
            "quality and internal consistency. Give your own estimate, then assign "
            "confidence using this exact format on separate lines: "
            "CONFIDENCE: <agent_name>: 0.XX  (0.0–1.0). "
            "End with BOTH lines, and make them agree:\n"
            "DIRECTION: UP or DIRECTION: DOWN or DIRECTION: FLAT\n"
            "ESTIMATE: +₱X.XX/L or ESTIMATE: -₱X.XX/L"
        )
    return base + (
        "End with BOTH lines, and make them agree:\n"
        "DIRECTION: UP or DIRECTION: DOWN or DIRECTION: FLAT\n"
        "ESTIMATE: +₱X.XX/L or ESTIMATE: -₱X.XX/L"
    )


def build_swarm_agents(current_price: float = _FALLBACK_RETAIL_PRICE_PHP) -> list[SwarmAgent]:
    """Build all 20 SwarmAgents (4 groups × 5 agents = 1 per role per group)."""
    agents: list[SwarmAgent] = []
    for group_id, region in enumerate(REGIONS):
        for role in _ROLE_ORDER:
            agents.append(SwarmAgent(
                name=f"{region} {role}",
                role=role,
                tier=_ROLE_TIERS[role],
                group_id=group_id,
                region_name=region,
                system_prompt=_make_system_prompt(role, region, current_price),
                rag_sources=_ROLE_RAG[role],
            ))
    return agents


# ── Scoring utilities ─────────────────────────────────────────────────────────

# Separator a model actually writes between the name and the number. The prompt
# asks for a colon; models routinely use a dash, an en dash, or an equals sign,
# and often wrap the label in markdown bold.
_SEP = r'[:\-–—=]'
_MD = r'[\*_`\s]*'


def _parse_scores(text: str, agent_names: list[str]) -> dict[str, float]:
    """Parse 'SCORE: <name>: X' lines. Missing agents default to 0.5.

    Deliberately tolerant. When this parser misses, EVERY agent falls back to 0.5,
    every combined score becomes exactly 0.50, and the elimination bracket stops
    measuring anything -- which is what a live run showed. Strictness here does
    not make the tournament more rigorous, it makes it silently random, so an
    "8/10" or a dash separator is accepted rather than discarded.
    """
    result = {}
    for name in agent_names:
        m = re.search(
            rf'SCORE{_MD}{_SEP}?{_MD}{re.escape(name)}{_MD}{_SEP}{_MD}'
            rf'(\d+(?:\.\d+)?)\s*(?:/\s*10)?',
            text, re.IGNORECASE)
        result[name] = min(float(m.group(1)), 10.0) / 10.0 if m else 0.5
    return result


def _parse_confidence(text: str, agent_names: list[str]) -> dict[str, float]:
    """Parse 'CONFIDENCE: <name>: 0.XX' lines. Missing agents default to 0.5.

    Same tolerance as `_parse_scores`, plus percentages: a model asked for 0.85
    frequently answers 85%.
    """
    result = {}
    for name in agent_names:
        m = re.search(
            rf'CONFIDENCE{_MD}{_SEP}?{_MD}{re.escape(name)}{_MD}{_SEP}{_MD}'
            rf'(\d+(?:\.\d+)?)\s*(%?)',
            text, re.IGNORECASE)
        if not m:
            result[name] = 0.5
            continue
        value = float(m.group(1))
        if m.group(2) == '%' or value > 1.0:      # 85% or a bare 85
            value /= 100.0
        result[name] = max(0.0, min(1.0, value))
    return result


def compute_combined_score(
    response: AgentResponse,
    critic_score: float,
    confidence: float,
    group_estimates: list[float],
) -> float:
    """
    combined = 0.4 × critic_score + 0.6 × (confidence × (1 − deviation_normalized))
    Returns 0.0 if response.price_estimate is None.
    """
    if response.price_estimate is None:
        return 0.0
    if len(group_estimates) < 2:
        deviation_norm = 0.0
    else:
        est_range = max(group_estimates) - min(group_estimates)
        median = sorted(group_estimates)[len(group_estimates) // 2]
        deviation_norm = (abs(response.price_estimate - median) / est_range
                          if est_range > 0 else 0.0)
        # Clamped because the score must never go negative. `deviation_norm`
        # exceeds 1 only when the scored estimate is not a member of
        # `group_estimates`, which the arena never does today — but an unguarded
        # negative would sort BELOW the 0.0 given to an agent that produced no
        # estimate at all, so a caller passing a mismatched list would silently
        # invert the elimination and cut the agents who answered.
        deviation_norm = min(deviation_norm, 1.0)
    return 0.4 * critic_score + 0.6 * (confidence * (1.0 - deviation_norm))


def scores_are_degenerate(agents: list[SwarmAgent], tol: float = 1e-9) -> bool:
    """True when the tournament failed to tell the agents apart.

    Worth surfacing rather than hiding: if every combined score is equal, the
    bracket is not selecting on quality, it is just removing whoever the tie-break
    happens to reach.
    """
    if len(agents) < 2:
        return False
    scores = [a.combined_score for a in agents]
    return (max(scores) - min(scores)) <= tol


def _tie_key(agent: SwarmAgent) -> int:
    """Role-neutral, reproducible tie-break.

    NOT the agent's position in the list. `alive` is ordered by `_ROLE_ORDER`, and
    Python's sort is stable, so a plain sort on a tie eliminated agents in role
    order: Forecaster first, every single run, leaving ConfidenceScorer as the
    winner by construction. That is a structural bias against exactly the role the
    product exists to run, and it fires whenever the Critic and ConfidenceScorer
    produce no parseable scores, which makes every combined score 0.50.

    Hashing the name keeps runs reproducible (ADR-002) without letting a hardcoded
    role list decide who survives.
    """
    return llm.derive_seed('tiebreak', agent.name)


def eliminate_bottom_n(
    agents: list[SwarmAgent], n: int
) -> tuple[list[SwarmAgent], list[SwarmAgent]]:
    """Sort by combined_score ascending; remove bottom n. Returns (survivors, eliminated).

    Ties are broken by a hash of the agent name rather than by list position, so
    elimination cannot be decided by where a role sits in `_ROLE_ORDER`.

    At least one agent always survives. `_BRACKET` removes 2 then 2, which assumes
    a group of exactly `len(_ROLE_ORDER)`. That assumption is not guaranteed:
    `evolved_agents` replaces the whole roster, so a group can arrive with fewer.
    Four eliminations from a group of four left the list empty and the arena then
    raised `IndexError: list index out of range` from `alive[0]`, with the
    traceback discarded by the per-group handler. Clamping here makes the bracket
    correct for any roster size instead of only the default one.
    """
    if not agents:
        return [], []
    keep_at_least = 1
    n = max(0, min(n, len(agents) - keep_at_least))
    sorted_agents = sorted(agents, key=lambda a: (a.combined_score, _tie_key(a)))
    eliminated = sorted_agents[:n]
    survivors = sorted_agents[n:]
    for e in eliminated:
        e.is_alive = False
    return survivors, eliminated


# ── GroupArena ────────────────────────────────────────────────────────────────

# Rounds: (round_number, agents_to_eliminate_this_round)
# 5 agents → eliminate 2 → 3 left → eliminate 2 → 1 winner

#: Agents that must still be in the room when the final round runs.
#:
#: A fixed "remove 2, then remove 2" assumes a roster of five. Evolution benches
#: agents, so groups arrive with three or four, and three minus two is a final
#: round of ONE AGENT TALKING TO ITSELF. That round produces a single estimate,
#: which is below the agreement quorum, so the group cannot be scored at all —
#: exactly how NCR dropped out of the run 28 headline.
_FINAL_ROUND_AGENTS = 3


def build_bracket(n_agents: int) -> list[tuple[int, int]]:
    """Rounds and eliminations sized for the roster that actually turned up.

    Round 1 trims to `_FINAL_ROUND_AGENTS` (or to the whole roster if it is
    already smaller), round 2 runs as a real debate and then resolves to one
    winner. On the default five-agent roster this reproduces the old fixed
    bracket exactly: remove 2, then remove 2.
    """
    if n_agents <= 1:
        return [(1, 0)]
    keep = min(n_agents, _FINAL_ROUND_AGENTS)
    return [(1, n_agents - keep), (2, keep - 1)]


#: The bracket for a full roster. Still a module attribute because the ablation
#: harness reshapes it to measure a one-round swarm; `_bracket_for` honours that
#: override whenever the roster can absorb it.
_BRACKET = build_bracket(len(_ROLE_ORDER))


def _bracket_for(agents: list) -> list[tuple[int, int]]:
    """The configured bracket when the roster can take it, else one that fits."""
    if sum(n for _, n in _BRACKET) <= len(agents) - 1:
        return _BRACKET
    return build_bracket(len(agents))


class GroupArena:
    def __init__(
        self,
        group_id: int,
        agents: list[SwarmAgent],
        rag: RagEngine,
        scenario: dict,
        on_event: Optional[Callable] = None,
        data_brief: Optional['LiveDataBrief'] = None,
        ml_baseline: str = '',
        anchor: Optional[float] = None,
        blind_round_one: bool = True,
        reconcile: bool = True,
    ):
        self._group_id = group_id
        self._agents = agents          # 5 SwarmAgents, all is_alive=True
        self._rag = rag
        self._scenario = scenario
        self._on_event = on_event      # callable(event_type, *args)
        self._data_brief = data_brief
        self._ml_baseline = ml_baseline
        self._anchor = anchor
        # Two switches for the mechanisms that can manufacture agreement.
        #
        # `blind_round_one` hides same-round peer statements from the estimating
        # roles, and is ON. An agent forming its opening read no longer sees what
        # its neighbours just said, which is what the Forum has always done and
        # the swarm never did. Measured over one paired run: the room went from
        # 2 distinct estimates to 6, and the spread from 0.26 to 2.50 PHP/L, at a
        # cost of 8 points of reported agreement. That trade is the point rather
        # than a regression — a lower number over six real opinions beats a higher
        # one over two, and the previous configuration was scoring the room's
        # collapse as its consensus.
        #
        # The Critic and ConfidenceScorer are exempt via `_PEER_READING_ROLES`:
        # they score other agents by name and a blind Critic has nothing to
        # critique. Round 1 therefore stays sequential, because those two still
        # need the estimating roles to have spoken first.
        #
        # `reconcile` controls the round-2 instruction to move toward the group
        # median, and stays ON. The blind-arm experiment could not separate its
        # effect from noise: removing it SCORED HIGHER (96 against 92), which it
        # cannot genuinely do, so that arm measured sampling variation rather
        # than the rule. It is left alone until there is a reason to touch it.
        self._blind_round_one = blind_round_one
        self._reconcile = reconcile
        self._history: list[AgentResponse] = []

    def _scenario_text(self) -> str:
        s = self._scenario
        return (
            f"Current PH retail gasoline baseline: ₱{s.get('current_price', _FALLBACK_RETAIL_PRICE_PHP):.2f}/L. "
            f"AUTHORITATIVE SCENARIO SHOCK: oil price {s.get('oil_pct', 0):+.1f}%, "
            f"USD/PHP {s.get('usd_pct', 0):+.1f}%, "
            f"BSP rate {s.get('bsp_rate', 6.5):.2f}%, "
            f"demand index {s.get('demand_index', 72):.0f}. "
            "Treat DATA BRIEF market history as calibration context, not as a replacement for this scenario."
        )

    def _brief_block(self) -> str:
        if self._data_brief is None:
            return ''
        try:
            return self._data_brief.as_prompt_block(self._scenario) + '\n\n'
        except Exception:
            return ''

    def _calibration_rule(self) -> str:
        """Output constraints, plus anchor guidance only when an anchor exists.

        This used to emit the anchor bullets unconditionally, falling back to the
        literal phrase "the ML anchor if supplied by the prompt". Agents were told
        to treat that sentence as their centre of gravity and to stay within
        +/-P1.00/L of it, which is not a rule so much as a dangling pointer. The
        format constraints below are always meaningful; the anchor ones are not.
        """
        always = (
            "- Do not output absolute pump prices. Output only the next price CHANGE.\n"
            f"- Any estimate outside +/-P{_MAX_REALISTIC_FUEL_CHANGE:.0f}/L is invalid.\n"
        )
        # getattr, not self._anchor: the prompt tests build an arena through
        # __new__ with only the fields the rule reads.
        anchor = getattr(self, '_anchor', None)
        # The ML baseline is a second opinion, not a second target. When both are
        # present the pass-through is the one to reason from — it is accounting,
        # the regressor is a fit — so it goes first and the ML figure is offered
        # as corroboration. Two competing "centres of gravity" in one prompt is
        # how a 3b model ends up splitting the difference between them and
        # calling it analysis.
        anchor_rule = (
            "\nPHYSICAL BASELINE:\n" + anchor_prompt_block(anchor) + "\n"
            if anchor is not None else ''
        )
        if not self._ml_baseline:
            if anchor_rule:
                return anchor_rule + "\nOUTPUT RULE:\n" + always
            return "\nOUTPUT RULE:\n" + always
        return (
            anchor_rule
            + "\nCALIBRATION RULE:\n"
            + (f"- A statistical model trained on PH history reads "
               f"{self._ml_baseline}. Treat it as corroboration of the physical "
               f"baseline above, not as a second target.\n"
               if anchor_rule else
               f"- Treat {self._ml_baseline} as the center of gravity for the forecast.\n"
               "- Your estimate should normally stay within +/-P1.00/L of that anchor.\n"
               "- You may leave that band only if you cite a specific DATA BRIEF figure "
               "or peer argument explaining why.\n")
            + always
        )

    def _reconciliation_rule(self) -> str:
        estimates = [
            r.price_estimate for r in self._history
            if _is_realistic_fuel_change(r.price_estimate)
        ]
        if not estimates:
            return ''
        median = statistics.median(estimates)
        low = min(estimates)
        high = max(estimates)
        return (
            "\nRECONCILIATION RULE:\n"
            f"- Prior valid estimate range: {low:+.2f} to {high:+.2f} P/L; "
            f"group median: {median:+.2f} P/L.\n"
            "- If your estimate differs from the group median by more than P1.00/L, "
            "revise toward the median or explicitly cite the reason for keeping the disagreement.\n"
            "- Prefer a calibrated consensus over a dramatic outlier unless the data clearly supports it.\n"
        )

    def _build_prompt(
        self,
        agent: SwarmAgent,
        round_num: int,
        round_responses: list[AgentResponse],
    ) -> list[dict]:
        scenario_text = self._scenario_text()
        chunks = self._rag.query(scenario_text, top_k=3, sources=agent.rag_sources)
        # Preserved verbatim on the response so the graph can show what was read
        # rather than re-deriving it later (RSK-019).
        self._last_retrieval = [
            {'source': c.get('source', '?'), 'text': c.get('text', '')}
            for c in (chunks or [])
        ]
        rag_text = '\n'.join(
            f"[{c['source']}] {c['text'][:200]}" for c in chunks
        ) or 'No context.'
        prior_rounds = '\n'.join(
            f"{r.agent_name} (Round {r.round_num}): {r.statement[:300]}"
            for r in self._history
        )
        this_round = '\n'.join(
            f"{r.agent_name}: {r.statement[:300]}"
            for r in round_responses
        )
        user_parts = [
            self._brief_block(),
            scenario_text,
            f"\nContext:\n{rag_text}",
            self._calibration_rule(),
        ]
        if self._ml_baseline:
            user_parts.append(f"\nML ANCHOR: {self._ml_baseline}\n"
                              "(Use this as a calibration anchor — debate around it, "
                              "not away from it without strong evidence. Estimates more than "
                              f"±{_MAX_REALISTIC_FUEL_CHANGE:.0f}/L are invalid absolute-price parses.)")
        if prior_rounds:
            # Seeing EARLIER rounds is the debate working: round 2 exists so an
            # agent can respond to what the room said. Only same-round peer
            # visibility is contamination, so `prior_rounds` is never hidden.
            user_parts.append(f"\nPrevious rounds:\n{prior_rounds}")
            if self._reconcile:
                user_parts.append(self._reconciliation_rule())
        # The Critic and the ConfidenceScorer score other agents BY NAME, so they
        # cannot work blind. The three estimating roles can, and theirs are the
        # estimates the agreement metric counts.
        if this_round and (not self._blind_round_one
                           or agent.role in _PEER_READING_ROLES):
            user_parts.append(f"\nThis round so far:\n{this_round}")
        user_parts.append(
            "\nYou MUST cite specific data from the DATA BRIEF when available. "
            "Give your analysis and end with ALL THREE lines. The DIRECTION and "
            "the sign of the ESTIMATE must agree with each other and with your "
            "causal chain:\n"
            "CAUSAL CHAIN: [scenario shock] -> [market effect] -> [retail mechanism] -> [consumer impact]\n"
            "DIRECTION: UP or DIRECTION: DOWN or DIRECTION: FLAT\n"
            "ESTIMATE: +₱X.XX/L or ESTIMATE: -₱X.XX/L"
        )
        return [
            {'role': 'system', 'content': agent.system_prompt},
            {'role': 'user', 'content': ''.join(user_parts)},
        ]

    def _call_agent(self, agent: SwarmAgent, messages: list[dict]) -> AgentResponse:
        if self._on_event:
            self._on_event('agent_typing', self._group_id, agent.name)
        full_text = ''
        seed = _vintage_seed(self._group_id, agent.name)
        for token in llm.stream(messages, tier=agent.tier,
                                max_tokens=_AGENT_MAX_TOKENS, seed=seed):
            full_text += token
        if self._on_event:
            self._on_event('agent_done_typing', self._group_id, agent.name)
        thinking, statement = _parse_think(full_text)
        estimate = _extract_fuel_change(statement)
        if estimate is None:
            logging.info('swarm group %s: %s gave no parseable estimate, re-asking',
                         self._group_id, agent.name)
            statement, estimate = _reask_for_estimate(
                messages, statement, tier=agent.tier,
                max_tokens=_AGENT_RETRY_MAX_TOKENS,
                seed=_vintage_seed(self._group_id, agent.name,
                                    'retry'),
            )
        # An agent that reasons one way and signs the other is the largest single
        # source of apparent disagreement measured on this swarm, so the number is
        # checked against the direction the agent itself stated.
        statement, estimate = _resolve_direction_conflict(
            messages, statement, parse_direction(statement), estimate,
            tier=agent.tier, max_tokens=_AGENT_RETRY_MAX_TOKENS,
            seed=_vintage_seed(self._group_id, agent.name, 'direction'),
        )
        return AgentResponse(
            agent_name=agent.name,
            round_num=0,
            thinking=thinking,
            statement=statement,
            price_estimate=estimate,
            retrieval=list(getattr(self, '_last_retrieval', [])),
        )

    def run(self) -> GroupSurvivor:
        alive = sorted(self._agents, key=lambda a: _ROLE_ORDER.index(a.role))

        for round_num, n_eliminate in _bracket_for(alive):
            if round_num == 1:
                # Round 1: sequential so each agent can read peers' responses in order.
                # Critic and ConfidenceScorer react to Forecaster/Synthesizer/Extractor
                # — the sequential context is what makes the debate meaningful.
                round_responses: list[AgentResponse] = []
                for agent in alive:
                    messages = self._build_prompt(agent, round_num, round_responses)
                    resp = self._call_agent(agent, messages)
                    # Carry `retrieval` across: rebuilding positionally dropped it
                    # and sent the graph back to reconstructing evidence (RSK-019).
                    resp = AgentResponse(agent.name, round_num, resp.thinking,
                                         resp.statement, resp.price_estimate,
                                         retrieval=list(resp.retrieval or []))
                    round_responses.append(resp)
                self._history.extend(round_responses)
            else:
                # Round 2+: agents already debated in Round 1; run in parallel.
                # Each agent sees the full Round 1 history via self._history.
                def _call_one(agent: SwarmAgent, rn: int = round_num) -> AgentResponse:
                    msgs = self._build_prompt(agent, rn, [])
                    resp = self._call_agent(agent, msgs)
                    return AgentResponse(agent.name, rn, resp.thinking,
                                         resp.statement, resp.price_estimate,
                                         retrieval=list(resp.retrieval or []))

                with concurrent.futures.ThreadPoolExecutor(max_workers=len(alive)) as pool:
                    futs = {pool.submit(_call_one, a): a for a in alive}
                    name_to_resp: dict[str, AgentResponse] = {}
                    for fut in concurrent.futures.as_completed(futs):
                        name_to_resp[futs[fut].name] = fut.result()

                round_responses = [name_to_resp[a.name] for a in alive]
                self._history.extend(round_responses)

            # Collect scores from Critic responses
            critic_responses = [
                r for r, a in zip(round_responses, alive) if a.role == 'Critic'
            ]
            alive_names = [a.name for a in alive]
            merged_critic_text = ' '.join(r.statement for r in critic_responses)
            critic_scores = _parse_scores(merged_critic_text, alive_names)

            # Collect confidence from ConfidenceScorer responses
            conf_responses = [
                r for r, a in zip(round_responses, alive) if a.role == 'ConfidenceScorer'
            ]
            merged_conf_text = ' '.join(r.statement for r in conf_responses)
            conf_scores = _parse_confidence(merged_conf_text, alive_names)

            # Collect group estimates for deviation calc
            group_estimates = [
                r.price_estimate for r in round_responses if r.price_estimate is not None
            ]

            # Assign combined scores
            for agent, resp in zip(alive, round_responses):
                agent.combined_score = compute_combined_score(
                    resp,
                    critic_score=critic_scores.get(agent.name, 0.5),
                    confidence=conf_scores.get(agent.name, 0.5),
                    group_estimates=group_estimates,
                )

            degenerate = scores_are_degenerate(alive)
            if degenerate:
                # Every agent scored identically, so this round removes agents
                # without measuring anything. Say so rather than let the bracket
                # look like a quality tournament.
                logging.warning(
                    'swarm group %s round %s: all combined scores equal (%.2f); '
                    'the Critic and ConfidenceScorer produced no parseable scores, '
                    'so this elimination is arbitrary',
                    self._group_id, round_num,
                    alive[0].combined_score if alive else float('nan'))
                if self._on_event:
                    self._on_event('scoring_degenerate', self._group_id, round_num,
                                   len(alive))

            alive, eliminated = eliminate_bottom_n(alive, n=n_eliminate)

            for e in eliminated:
                if self._on_event:
                    self._on_event('eliminated', self._group_id, e.name,
                                   e.combined_score, round_num)

            if self._on_event:
                self._on_event('group_round_done', self._group_id, round_num,
                               list(round_responses))

        if not alive:
            # Unreachable now that eliminate_bottom_n keeps one, but stated so a
            # future bracket change fails with a diagnosable message rather than a
            # bare IndexError from a worker thread.
            raise RuntimeError(
                f'swarm group {self._group_id}: the bracket eliminated every agent; '
                f'started with {len(self._agents)} and the bracket removes '
                f'{sum(n for _, n in _bracket_for(self._agents))}')
        winner = alive[0]
        winner_resp = next(
            (r for r in reversed(self._history) if r.agent_name == winner.name), None)
        if winner_resp is None:
            raise RuntimeError(
                f'swarm group {self._group_id}: winner {winner.name!r} has no response '
                f'in a history of {len(self._history)} turns')
        survivor = GroupSurvivor(
            group_id=self._group_id,
            region_name=winner.region_name,
            response=winner_resp,
            combined_score=winner.combined_score,
            agent_role=winner.role,
            # Record the concrete model, not the tier: this is provenance for
            # the report, and 'fast' resolves differently per provider/config.
            agent_model=llm.describe_model(winner.tier),
        )
        if self._on_event:
            self._on_event('survivor', self._group_id, survivor)
        return survivor


# ── RegionalJudge ─────────────────────────────────────────────────────────────

class RegionalJudge:
    def __init__(
        self,
        judge_id: int,
        survivors: tuple[GroupSurvivor, GroupSurvivor],
        rag: RagEngine,
        scenario: dict,
        data_brief: Optional['LiveDataBrief'] = None,
        agent_estimates: Optional[list[float]] = None,
        anchor: Optional[float] = None,
    ):
        self._judge_id = judge_id
        self._s1, self._s2 = survivors
        self._rag = rag
        self._scenario = scenario
        self._data_brief = data_brief
        # Every estimate produced by the agents in this judge's two regions.
        # Without them there is nothing to measure agreement over — see run().
        self._agent_estimates = agent_estimates or []
        self._anchor = anchor

    def _anchor_block(self) -> str:
        """The physical baseline, or nothing when the caller supplied none.

        A regional judge resolving two estimates ₱7 apart has no way to tell
        which one is the right scale unless it is told what the scale is.
        """
        if self._anchor is None:
            return ''
        return anchor_prompt_block(self._anchor) + '\n\n'

    def _brief_block(self) -> str:
        if self._data_brief is None:
            return ''
        try:
            return self._data_brief.as_prompt_block(self._scenario) + '\n\n'
        except Exception:
            return ''

    def _scenario_text(self) -> str:
        s = self._scenario
        return (
            f"Current PH retail gasoline baseline: ₱{s.get('current_price', _FALLBACK_RETAIL_PRICE_PHP):.2f}/L. "
            f"AUTHORITATIVE SCENARIO SHOCK: oil {s.get('oil_pct', 0):+.1f}%, "
            f"USD/PHP {s.get('usd_pct', 0):+.1f}%, "
            f"BSP {s.get('bsp_rate', 6.5):.2f}%, "
            f"demand {s.get('demand_index', 72):.0f}. "
            "Treat DATA BRIEF market history as calibration context, not as a replacement for this scenario."
        )

    def _defense_prompt(
        self, defender: GroupSurvivor, opponent: GroupSurvivor
    ) -> list[dict]:
        return [
            {'role': 'system', 'content': (
                f"You are a regional economic analyst representing the {defender.region_name} "
                "region. Defend your price estimate against your opponent's critique. "
                "Cite DATA BRIEF figures when available. "
                f"Ignore estimates outside ±{_MAX_REALISTIC_FUEL_CHANGE:.0f}/L as invalid absolute-price parses. "
                "Apply the project calibration policy: prefer estimates close to the group median unless a cited figure justifies disagreement. "
                "End with BOTH lines, and make them agree:\n"
            "DIRECTION: UP or DIRECTION: DOWN or DIRECTION: FLAT\n"
            "ESTIMATE: +₱X.XX/L or ESTIMATE: -₱X.XX/L"
            )},
            {'role': 'user', 'content': (
                f"{self._brief_block()}{self._scenario_text()}\n\n"
                f"{self._anchor_block()}"
                f"Your previous estimate: {defender.response.statement[:400]}\n\n"
                f"Opponent ({opponent.region_name}) argues: {opponent.response.statement[:400]}\n\n"
                "Defend your position or update your estimate based on their critique."
            )},
        ]

    def _synthesis_prompt(
        self, defense1: str, defense2: str
    ) -> list[dict]:
        return [
            {'role': 'system', 'content': (
                "You are a regional judge synthesizing two regional estimates into a "
                "single consensus. Weigh both defenses, resolve differences, and produce "
                "a final regional verdict. "
                "Cite DATA BRIEF figures when available. "
                f"Ignore estimates outside ±{_MAX_REALISTIC_FUEL_CHANGE:.0f}/L as invalid absolute-price parses. "
                "Apply the project reconciliation policy: prefer the calibrated midpoint unless a cited figure justifies a regional exception. "
                "End with BOTH lines, and make them agree:\n"
            "DIRECTION: UP or DIRECTION: DOWN or DIRECTION: FLAT\n"
            "ESTIMATE: +₱X.XX/L or ESTIMATE: -₱X.XX/L"
            )},
            {'role': 'user', 'content': (
                f"{self._brief_block()}{self._scenario_text()}\n\n"
                f"{self._anchor_block()}"
                f"{self._s1.region_name} defense: {defense1[:500]}\n\n"
                f"{self._s2.region_name} defense: {defense2[:500]}\n\n"
                "Produce the final regional consensus estimate."
            )},
        ]

    def _call(self, messages: list[dict], tier: str = _JUDGE_TIER,
              tag: str = 'judge') -> str:
        # `tag` distinguishes the three prompts this method serves; without it all
        # three would share a seed and the judge would answer itself identically.
        seed = _vintage_seed(self._judge_id, tag)
        full = ''.join(llm.stream(messages, tier=tier,
                                  max_tokens=_JUDGE_MAX_TOKENS, seed=seed))
        _, statement = _parse_think(full)
        return statement

    def run(self) -> RegionalVerdict:
        def1 = self._call(self._defense_prompt(self._s1, self._s2), tag='defense1')
        def2 = self._call(self._defense_prompt(self._s2, self._s1), tag='defense2')
        synthesis_messages = self._synthesis_prompt(def1, def2)
        synthesis = self._call(synthesis_messages, tag='synthesis')
        estimate, rejected = parse_fuel_estimate(synthesis)
        if estimate is None and rejected is None:
            # The judge stated no number at all, which prints "no estimate" on the
            # card. Distinct from `rejected`, where it gave one and the guard threw
            # it away: that is a judgement worth reporting, this is a lost turn
            # after three deep-tier calls. Re-ask once before spending them again.
            logging.info('swarm judge %s: synthesis had no estimate, re-asking',
                         self._judge_id)
            synthesis, estimate = _reask_for_estimate(
                synthesis_messages, synthesis, tier=_JUDGE_TIER,
                max_tokens=_JUDGE_RETRY_MAX_TOKENS,
                seed=_vintage_seed(self._judge_id,
                                    'synthesis', 'retry'),
            )
        # The judges need this at least as much as the agents. A live run had
        # Western Visayas and Davao return +1.00 against a -3.28 anchor at 97
        # percent internal agreement: the verdict a whole half of the country was
        # reported on, inverted, and confident enough to read as consensus.
        synthesis, estimate = _resolve_direction_conflict(
            synthesis_messages, synthesis, parse_direction(synthesis), estimate,
            tier=_JUDGE_TIER, max_tokens=_JUDGE_RETRY_MAX_TOKENS,
            seed=_vintage_seed(self._judge_id, 'synthesis', 'direction'),
        )
        # Measured, not assumed. This was previously a hardcoded 0.75 whenever
        # the estimate merely parsed, which the report then displayed as "agent
        # agreement" — a constant presented as a measurement, and identical on
        # every card. Uses the same function as the master verdict so the word
        # "agreement" means one thing everywhere in the report.
        confidence = _robust_confidence_pct(self._agent_estimates, estimate) / 100
        return RegionalVerdict(
            judge_id=self._judge_id,
            region_pair=(self._s1.region_name, self._s2.region_name),
            estimate=estimate,
            confidence=confidence,
            reasoning=synthesis,
            survivor_names=(self._s1.response.agent_name, self._s2.response.agent_name),
            rejected_estimate=rejected,
        )


# ── MasterJudge ───────────────────────────────────────────────────────────────

_HIGH_WEIGHT_REGIONS = {'NCR'}


class MasterJudge:
    def __init__(
        self,
        verdicts: list[RegionalVerdict],
        rag: RagEngine,
        scenario: dict,
        survivors: Optional[list[GroupSurvivor]] = None,
        data_brief: Optional['LiveDataBrief'] = None,
        group_histories: Optional[dict[int, list]] = None,
        anchor: Optional[float] = None,
    ):
        self._verdicts = verdicts
        self._rag = rag
        self._scenario = scenario
        self._survivors = survivors or []
        self._data_brief = data_brief
        # Every agent response, per group. Agreement is measured over these —
        # see run(). Without them the master can only see the survivors, which
        # is how the headline came to be a two-agent measurement.
        self._group_histories = group_histories or {}
        self._anchor = (anchor if anchor is not None
                        else self._compute_physical_anchor())

    def _compute_physical_anchor(self) -> float:
        return compute_physical_anchor(self._scenario, self._data_brief)

    def _brief_block(self) -> str:
        if self._data_brief is None:
            return ''
        try:
            return self._data_brief.as_prompt_block(self._scenario) + '\n\n'
        except Exception:
            return ''

    def _build_prompt(self) -> list[dict]:
        s = self._scenario
        scenario_text = (
            f"Current PH retail gasoline baseline: ₱{s.get('current_price', _FALLBACK_RETAIL_PRICE_PHP):.2f}/L. "
            f"AUTHORITATIVE SCENARIO SHOCK: oil {s.get('oil_pct', 0):+.1f}%, "
            f"USD/PHP {s.get('usd_pct', 0):+.1f}%, "
            f"BSP {s.get('bsp_rate', 6.5):.2f}%, demand {s.get('demand_index', 72):.0f}. "
            "Treat DATA BRIEF market history as calibration context, not as a replacement for this scenario."
        )
        verdicts_text = '\n\n'.join(
            f"[{'HIGH WEIGHT — ' if any(r in _HIGH_WEIGHT_REGIONS for r in v.region_pair) else ''}"
            f"{' & '.join(v.region_pair)}] "
            f"Estimate: {f'+₱{v.estimate:.2f}/L' if v.estimate is not None else 'N/A'} "
            f"(confidence {v.confidence:.2f})\n{v.reasoning[:400]}"
            for v in self._verdicts
        )
        # The same block the agents and the regional judges now see, plus the
        # national-scale note that only applies to the headline number.
        anchor_text = (
            anchor_prompt_block(self._anchor)
            + " A national monthly pump change is almost never more than ~₱2/L "
              "away from this number."
        )
        return [
            {'role': 'system', 'content': (
                "You are the Master Judge synthesizing 2 regional Philippine fuel price "
                "estimates into a single national verdict. Give special weight to the "
                "NCR region as it represents the majority of fuel consumption. "
                "Anchor your number to the MECHANICAL PASS-THROUGH provided and adjust "
                "within a narrow band; do not restate a regional outlier that ignores it. "
                "Cite DATA BRIEF figures when available. Identify any dissenting regions. "
                f"Ignore estimates outside ±{_MAX_REALISTIC_FUEL_CHANGE:.0f}/L as invalid absolute-price parses. "
                "End with BOTH lines, and make them agree:\n"
            "DIRECTION: UP or DIRECTION: DOWN or DIRECTION: FLAT\n"
            "ESTIMATE: +₱X.XX/L or ESTIMATE: -₱X.XX/L"
            )},
            {'role': 'user', 'content': (
                f"{self._brief_block()}{scenario_text}\n\n{anchor_text}\n\n"
                f"Regional verdicts:\n{verdicts_text}"
            )},
        ]

    def run(self) -> MasterVerdict:
        full = ''.join(
            llm.stream(self._build_prompt(), tier=_JUDGE_TIER,
                       max_tokens=_MASTER_MAX_TOKENS,
                       seed=_vintage_seed('master'))
        )
        _, statement = _parse_think(full)

        # Physics-anchored reconciliation: the weak model supplies direction and
        # qualitative judgement, the pass-through formula supplies scale. Feed it
        # the model's raw number even when that number failed the ±₱8 guard — the
        # anchor is a tighter, physics-based leash, so a +₱12.93/L reading is
        # clamped back toward physics (keeping its upward direction) rather than
        # discarded to a blank. Only a genuinely absent number falls back to the
        # anchor outright.
        accepted, rejected = parse_fuel_estimate(statement)
        model_estimate = accepted if accepted is not None else rejected
        reconciled = anchoring.reconcile_estimate(model_estimate, self._anchor)
        final_estimate = reconciled.value

        # Agreement is measured WITHIN a region group, over that group's AGENTS,
        # then averaged across the groups that met quorum.
        #
        # Two things have to be true at once, and the previous version got the
        # first right and the second badly wrong.
        #
        # Right: the average has to be per-region. Pooling every estimate counted
        # a designed difference as a disagreement — regions carry a freight
        # multiplier (NCR 1.00, Ilocos 1.05, and so on), so a Davao estimate is
        # SUPPOSED to differ from an NCR one, and folding both into one spread
        # reports geography as analytical conflict. Cross-region disagreement is
        # not lost either way: it is reported separately as `dissenting`.
        #
        # Wrong: it averaged over SURVIVORS. One survivor per group and two groups
        # per judge meant the headline was computed from exactly two numbers, and
        # when one of those two had no parseable estimate the whole pair was
        # dropped — leaving the national figure resting on a single pair of
        # agents. Run 28 published "38% agent agreement" measured over the Western
        # Visayas Critic and the Davao survivor while its own regional cards, which
        # read the full histories, showed 74% and 53% directly underneath. 38 is
        # also the floor of this metric for any two estimates ₱3.00 or more apart,
        # so the headline was pinned to a constant and reported as a measurement.
        #
        # Now: the same population the regional cards use — every agent estimate
        # in the group, all rounds — grouped by region rather than by judge pair.
        histories = self._group_histories or {}
        per_region: list[int] = []
        counted = 0
        echoed = 0
        scored_responses: list = []
        for _group_id, history in sorted(histories.items()):
            estimates = agent_estimates_of(history)
            if len(estimates) >= _AGREEMENT_QUORUM:
                per_region.append(_robust_confidence_pct(estimates, None))
                counted += len(estimates)
                scored_responses.extend(agent_responses_of(history))
                # Counted only over the estimates the percentage was measured on,
                # so the two numbers describe the same population.
                echoed += sum(1 for e in estimates
                              if abs(e - self._anchor) <= _ANCHOR_ECHO_TOLERANCE)

        if per_region:
            confidence_pct = int(round(statistics.mean(per_region)))
            agreement_n = counted
            agreement_regions = (len(per_region), len(histories))
        else:
            # No group produced two usable estimates on its own. Pool the agents
            # across regions instead — worse, because it reads the freight
            # multiplier as conflict, but still a measurement of the agents, and
            # `agreement_regions` of (0, n) is what tells the report to say the
            # per-region split was unavailable.
            #
            # It deliberately does NOT fall back to the survivors and the judge
            # verdicts, which is what it used to do. That path scored two judge
            # outputs against each other and published the result as "agent
            # agreement" — the same two-item measurement this whole change exists
            # to stop, just one layer further up.
            pooled = [e for h in histories.values() for e in agent_estimates_of(h)]
            confidence_pct, agreement_n = measure_agreement(pooled)
            agreement_regions = (0, len(histories))
            echoed = sum(1 for e in pooled
                         if _is_realistic_fuel_change(e)
                         and abs(e - self._anchor) <= _ANCHOR_ECHO_TOLERANCE)
            scored_responses = [r for h in histories.values()
                                for r in agent_responses_of(h)]

        # `agreement_distinct` counts over exactly the population the percentage
        # was measured on, so a reader comparing the two is comparing like with
        # like. `agreement_diversity` deliberately narrows to one statement per
        # agent: it asks whether the room's opening READ was independent, and an
        # agent cannot answer that twice.
        scored_estimates = [r.price_estimate for r in scored_responses]
        agreement_distinct = len({round(e, 2) for e in scored_estimates})
        agreement_diversity = opening_diversity(scored_responses)

        dissenting = [
            ' & '.join(v.region_pair)
            for v in self._verdicts
            if _is_realistic_fuel_change(v.estimate) and _is_realistic_fuel_change(final_estimate)
            and abs(v.estimate - final_estimate) > 0.50
        ]

        # Build per-region estimates using group survivor anchors
        anchor_estimates = {
            s.group_id: s.response.price_estimate
            for s in self._survivors
            if _is_realistic_fuel_change(s.response.price_estimate)
        }
        regional_estimates = derive_regional_estimates(final_estimate, anchor_estimates)

        return MasterVerdict(
            final_estimate=final_estimate,
            confidence_pct=confidence_pct,
            agreement_n=agreement_n,
            agreement_regions=agreement_regions,
            agreement_echo_n=echoed,
            agreement_distinct=agreement_distinct,
            agreement_diversity=agreement_diversity,
            dissenting_regions=dissenting,
            reasoning=statement,
            regional_verdicts=self._verdicts,
            regional_estimates=regional_estimates,
            physical_anchor=reconciled.anchor,
            estimate_source=reconciled.source,
        )


# ── SwarmOrchestrator ─────────────────────────────────────────────────────────

class SwarmOrchestrator:
    def __init__(
        self,
        rag: RagEngine,
        scenario: dict,
        parallel_n: int = 4,
        on_event: Optional[Callable] = None,
        data_brief: Optional['LiveDataBrief'] = None,
        ml_baseline: str = '',
        evolved_agents: Optional[list] = None,
        # Shipped defaults. See GroupArena for the measurement behind them.
        blind_round_one: bool = True,
        reconcile: bool = True,
    ):
        self._rag = rag
        self._scenario = scenario
        self._parallel_n = parallel_n
        self._on_event = on_event
        self._data_brief = data_brief
        self._ml_baseline = ml_baseline
        self._evolved_agents = evolved_agents
        self._blind_round_one = blind_round_one
        self._reconcile = reconcile

    def run(self) -> MasterVerdict:
        # The prompts get a price either way — a slightly stale baseline costs
        # the agents almost nothing. Grading is the caller that must know whether
        # it was real, so the liveness travels back on the verdict rather than
        # being inferred later from the value.
        live_price, price_is_live = fetch_live_retail_price_checked()
        self._scenario = {**self._scenario, 'current_price': live_price}
        # Computed once, given to everyone who produces a number: the agents, the
        # regional judges, and the master. It used to reach only the last of the
        # three, so agreement was measured across a room that had never been told
        # the scale it was estimating on.
        anchor = compute_physical_anchor(self._scenario, self._data_brief)
        if self._evolved_agents is not None:
            all_agents = self._evolved_agents
        else:
            all_agents = build_swarm_agents(live_price)
        sem = threading.Semaphore(self._parallel_n)
        # Derived from the agents actually built, not a hardcoded 4: the group
        # count follows REGIONS, so a hardcoded literal silently drops any
        # extra region and leaves a None survivor if one is removed.
        n_groups = len({a.group_id for a in all_agents})
        survivors: list[Optional[GroupSurvivor]] = [None] * n_groups
        errors: list[str] = []
        lock = threading.Lock()
        all_arena_responses: list = []
        # Kept per group, not just pooled, so each regional judge can measure
        # agreement across exactly the agents it is judging.
        group_histories: dict[int, list] = {}

        def run_group(group_id: int):
            with sem:
                group_agents = [a for a in all_agents if a.group_id == group_id]
                arena = GroupArena(
                    group_id=group_id,
                    agents=group_agents,
                    rag=self._rag,
                    scenario=self._scenario,
                    on_event=self._on_event,
                    data_brief=self._data_brief,
                    ml_baseline=self._ml_baseline,
                    anchor=anchor,
                    blind_round_one=self._blind_round_one,
                    reconcile=self._reconcile,
                )
                try:
                    s = arena.run()
                    with lock:
                        survivors[group_id] = s
                        group_histories[group_id] = list(arena._history)
                        all_arena_responses.extend(arena._history)
                except Exception as e:
                    # Keep the traceback. "Group 0: list index out of range" with
                    # no location is not a diagnosable error report, and this runs
                    # on a worker thread where the traceback is otherwise lost.
                    logging.exception('swarm group %s failed', group_id)
                    tb = traceback.extract_tb(e.__traceback__)
                    where = f' at {tb[-1].filename.split("/")[-1]}:{tb[-1].lineno}' if tb else ''
                    with lock:
                        errors.append(f"Group {group_id}: {type(e).__name__}: {e}{where}")

        threads = [threading.Thread(target=run_group, args=(i,))
                   for i in range(n_groups)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if errors:
            raise RuntimeError(f"Group errors: {'; '.join(errors)}")

        # Phase 2: regional judges (sequential)
        regional_verdicts: list[RegionalVerdict] = []
        for judge_id, (i, j) in enumerate(REGION_PAIRS):
            # A pair can reference a group that does not exist when REGIONS is
            # trimmed (e.g. during an ablation) — skip rather than IndexError.
            if i >= n_groups or j >= n_groups:
                continue
            s1, s2 = survivors[i], survivors[j]
            if s1 is None or s2 is None:
                continue
            judge = RegionalJudge(
                judge_id=judge_id,
                survivors=(s1, s2),
                rag=self._rag,
                scenario=self._scenario,
                data_brief=self._data_brief,
                agent_estimates=[
                    r.price_estimate
                    for gid in (i, j)
                    for r in group_histories.get(gid, [])
                    if r.price_estimate is not None
                ],
                anchor=anchor,
            )
            verdict = judge.run()
            regional_verdicts.append(verdict)
            if self._on_event:
                self._on_event('regional_done', judge_id, verdict)

        # Phase 3: master judge
        valid_survivors = [s for s in survivors if s is not None]
        master = MasterJudge(
            verdicts=regional_verdicts,
            rag=self._rag,
            scenario=self._scenario,
            survivors=valid_survivors,
            data_brief=self._data_brief,
            group_histories=group_histories,
            anchor=anchor,
        )
        mv = master.run()
        mv.all_responses = all_arena_responses
        # Carry the price the run reasoned from back to the caller, so the stored
        # scenario matches what the agents were told and grading has a real
        # baseline to measure against.
        #
        # None when the fetch failed. The caller then stores NO baseline rather
        # than the fallback, and grading skips the run instead of measuring a
        # change against a constant. That is the case that produced ten stored
        # grades of exactly +0.00: the fallback minus itself, which reads as a
        # quiet week and scores every estimate as wrong by its own magnitude.
        mv.current_price = live_price if price_is_live else None
        return mv


# ── SwarmThread ───────────────────────────────────────────────────────────────

class SwarmThread(QThread):
    group_round_done  = pyqtSignal(int, int, object)
    group_eliminated  = pyqtSignal(int, str, float, int)
    group_survivor    = pyqtSignal(int, object)
    agent_typing      = pyqtSignal(int, str)   # group_id, agent_name
    agent_done_typing = pyqtSignal(int, str)   # group_id, agent_name
    regional_done     = pyqtSignal(int, object)
    swarm_complete    = pyqtSignal(object)
    error_occurred    = pyqtSignal(str)

    def __init__(self, rag: RagEngine, scenario: dict, parallel_n: int = 4,
                 data_brief: Optional['LiveDataBrief'] = None,
                 ml_baseline: str = '', evolved_agents=None, parent=None):
        super().__init__(parent)
        self._rag = rag
        self._scenario = scenario
        self._parallel_n = parallel_n
        self._data_brief = data_brief
        self._ml_baseline = ml_baseline
        self._evolved_agents = evolved_agents

    def run(self):
        # PyQt6 routes signals emitted from non-QThread Python threads through
        # Qt's queued connection mechanism automatically — cross-thread emit is safe.
        def on_event(event_type, *args):
            if event_type == 'eliminated':
                self.group_eliminated.emit(*args)
            elif event_type == 'survivor':
                self.group_survivor.emit(*args)
            elif event_type == 'regional_done':
                self.regional_done.emit(*args)
            elif event_type == 'group_round_done':
                self.group_round_done.emit(*args)
            elif event_type == 'agent_typing':
                self.agent_typing.emit(*args)
            elif event_type == 'agent_done_typing':
                self.agent_done_typing.emit(*args)

        orch = SwarmOrchestrator(
            rag=self._rag,
            scenario=self._scenario,
            parallel_n=self._parallel_n,
            on_event=on_event,
            data_brief=self._data_brief,
            ml_baseline=self._ml_baseline,
            evolved_agents=self._evolved_agents,
        )
        try:
            mv = orch.run()
            self.swarm_complete.emit(mv)
        except Exception as e:
            self.error_occurred.emit(f"{type(e).__name__}: {e}")
