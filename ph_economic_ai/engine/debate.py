import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from PyQt6.QtCore import QThread, pyqtSignal

from ph_economic_ai.engine import llm
from ph_economic_ai.engine.rag import RagEngine
from ph_economic_ai.engine.live_data import LiveDataBrief


# The debate path runs far fewer calls than the swarm, so its main agents can
# afford the deep tier; the mini agents stay fast.
_MAIN_TIER = llm.DEEP
_MINI_TIER = llm.FAST

# ── Current Philippine economic baselines ─────────────────────────────────────
# Gas: see swarm.fetch_live_retail_price() — auto-fetches on every swarm run.
# Food and electricity: update these when PSA/Meralco publishes new figures.
_FOOD_INFLATION_YOY_PCT: float = 5.3      # PSA July 2026 (released 2026-08-05), year-on-year
_RICE_PRICE_PHP_KG: float = 52.0          # Commercial rice, NCR avg
_MERALCO_RATE_PHP_KWH: float = 14.8261   # Meralco residential July 2026
_GAS_PRICE_PHP_L: float = 98.82           # NCR Unleaded 91 avg May 20 2026


#: The slot names this codebase puts inside brackets when it shows an agent the
#: shape of a causal chain. Matched only INSIDE brackets and only against these
#: exact words, so an agent that writes its own bracketed aside — "[Brent +5%]"
#: — is not punished for using a bracket.
# -- The output lines every prompt in this project must use --------------------
#
# Instructions, NEVER templates. `engine.forum` found this first: with the
# wording "ESTIMATE: +PHP X.XX/L" a live 17-agent run had agents answer
# "ESTIMATE: +PHP X.XX/L", placeholder and all, which parses to nothing and was
# losing most of the roster's estimates. The swarm then reproduced the same
# failure twice more, in its causal-chain line and in its DIRECTION menu.
#
# Centralised here, in the layer `swarm` and `forum` both import, because the
# previous fix corrected one call site and left the agent system prompts, both
# judge layers and every persona in this file still handing models something to
# copy. Four discoveries of one bug is enough to give it a single home.
ESTIMATE_LINE = {
    'gas': ('ESTIMATE: <the peso-per-litre CHANGE you expect, signed> '
            '(worked example: "ESTIMATE: +0.85/L" for a rise of 85 centavos, or '
            '"ESTIMATE: -0.40/L" for a fall. Write your own number; never write '
            'X.XX.)'),
    'food': ('ESTIMATE: <the percent month-on-month CHANGE you expect, signed> '
             '(worked example: "ESTIMATE: +0.4%" or "ESTIMATE: -0.2%". Write '
             'your own number; never write X.X.)'),
    'electricity': ('ESTIMATE: <the peso-per-kWh CHANGE you expect, signed> '
                    '(worked example: "ESTIMATE: +0.30/kWh" or '
                    '"ESTIMATE: -0.15/kWh". Write your own number; never write '
                    'X.XX.)'),
}

#: The Critic's and ConfidenceScorer's per-agent ratings. These were
#: "SCORE: <agent_name>: X" and "CONFIDENCE: <agent_name>: 0.XX", the same
#: copyable shape as every other line this project has had to rewrite. They were
#: missed because the class-level guard checked a LIST OF KNOWN BAD STRINGS
#: rather than the property, so it could only ever catch what had already been
#: found.
#:
#: Likely the cause of Q-ENG-013. When a scorer echoes the template,
#: `_parse_scores` finds no real agent name, every agent falls back to 0.5, every
#: combined score comes out identical, and the elimination removes agents without
#: measuring anything. Seen live on 2026-07-31.
SCORE_LINE = ("SCORE: <the agent's name>: <your rating from 1 to 10> "
              '(worked example: "SCORE: NCR Forecaster: 7". Use each agent'
              "'s real name and your own number; never write agent_name or X.)")
CONFIDENCE_LINE = ("CONFIDENCE: <the agent's name>: <your confidence from 0.0 to "
                   '1.0> (worked example: "CONFIDENCE: NCR Forecaster: 0.75". '
                   "Use each agent's real name and your own number; never write "
                   "agent_name or 0.XX.)")

#: A menu is a template too. Five of twenty agents copied
#: "DIRECTION: UP or DIRECTION: DOWN or DIRECTION: FLAT" back verbatim.
DIRECTION_INSTRUCTION = (
    'DIRECTION: <write UP, DOWN or FLAT, the one you mean and not the list>')

_SCAFFOLD_SLOTS = frozenset({
    'scenario shock', 'market effect', 'retail mechanism', 'consumer impact',
    'trigger', 'price mechanism', 'household impact', 'effect',
})
_BRACKETED = re.compile(r'[\[<]\s*([^\]>\n]{1,40}?)\s*[\]>]')

_SCAFFOLD_RETRY_PROMPT = (
    'You returned the FORMAT of the causal chain instead of filling it in. The '
    'square brackets were placeholders describing what belongs in each link, not '
    'text to copy.\n'
    'Do not repeat your analysis. Reply with one CAUSAL CHAIN line and nothing '
    'else, replacing every link with the specific thing you mean — name the '
    'shock, the market it moves, the mechanism that reaches the pump, and the '
    'effect on a household. No brackets.\n'
    'CAUSAL CHAIN: ... -> ... -> ... -> ...'
)


def unfilled_scaffold(statement: str) -> bool:
    """True when an agent echoed the requested format instead of answering in it.

    Measured on 2026-07-29: eleven of twenty agents opened with

        CAUSAL CHAIN: [scenario shock] -> [market effect] -> [retail mechanism

    placeholders intact. `qwen2.5:3b` is small enough to treat the template as
    the answer. Nothing downstream noticed. The estimate still parsed, so the
    agent stayed in the agreement population contributing a number it had not
    reasoned to — and those are exactly the agents with nothing to offer but the
    anchor they were handed, which is the shape of the echo problem.

    It also corrupts `opening_diversity`, which reads these identical openings as
    a room herding when it is really a room not answering. Two different
    failures wearing the same number.

    Only the causal-chain line is inspected. A placeholder anywhere else is
    sloppy; a placeholder here means that link was never reasoned.
    """
    for raw in (statement or '').splitlines():
        marker = raw.upper().find('CAUSAL CHAIN')
        if marker < 0:
            continue
        # From the marker onward only. Prose can precede the line, and an agent
        # discussing "the [market effect]" before stating its chain has still
        # stated its chain.
        for slot in _BRACKETED.findall(raw[marker:]):
            if slot.strip().lower() in _SCAFFOLD_SLOTS:
                return True
    return False



@dataclass
class Agent:
    name: str
    role: str
    system_prompt: str
    rag_sources: list[str]
    tier: str = _MAIN_TIER      # llm.FAST | llm.DEEP — resolved at call time
    is_mini: bool = False       # True → smaller circle, lightweight model


@dataclass
class AgentResponse:
    agent_name: str
    round_num: int
    thinking: str
    statement: str
    price_estimate: Optional[float]  # ₱/L change; None if not parseable
    # What THIS agent retrieved for THIS turn, captured at generation time.
    #
    # RSK-019: the knowledge graph used to rebuild its evidence edges by calling
    # rag.query again when the graph was drawn. That answers "what would this agent
    # retrieve now", not "what did it read", and the two diverge whenever the
    # corpus, the embeddings, or top_k change. A graph that shows evidence the
    # agent never saw is worse than one that shows none, because it looks like
    # provenance.
    retrieval: list = field(default_factory=list)


DEFAULT_AGENTS: list[Agent] = [
    # ── 10 main agents (deep tier) ───────────────────────────────────────────
    Agent(
        name='Market Analyst',
        role='Price signals, news, short-term pass-through',
        system_prompt=(
            'You are a market analyst specializing in Philippine fuel markets. '
            'Using the provided news and price data, estimate the short-term '
            'retail gasoline price CHANGE (not absolute price) of the given scenario. '
            'End your response with exactly one line in this format: '
            + ESTIMATE_LINE['gas']
        ),
        rag_sources=['YahooFinanceCrude', 'YahooFinanceForex', 'BusinessWorld'],
    ),
    Agent(
        name='Policy Expert',
        role='Monetary policy, FX transmission, regulatory context',
        system_prompt=(
            'You are a policy expert focused on BSP monetary policy and peso dynamics. '
            'Using BSP statements and economic reports, challenge or support the '
            'previous estimate. End your response with exactly one line in this format: '
            + ESTIMATE_LINE['gas']
        ),
        rag_sources=['neda_2024_2026', 'YahooFinanceForex'],
    ),
    Agent(
        name='Risk Assessor',
        role='Tail risks, remittances, demand shocks, supply gaps',
        system_prompt=(
            'You are a risk assessor. Identify tail risks and softening factors '
            'the other agents may have missed. End your response with exactly one line: '
            + ESTIMATE_LINE['gas']
        ),
        rag_sources=['ManilaBulletin', 'neda_2024_2026'],
    ),
    Agent(
        name='Macroeconomist',
        role='GDP growth, inflation, interest rate transmission to fuel demand',
        system_prompt=(
            'You are a macroeconomist analyzing how Philippine GDP growth and '
            'inflation dynamics affect fuel demand and retail price pass-through. '
            'Factor in second-order effects like transportation cost inflation. '
            'End with: ' + ESTIMATE_LINE['gas']
        ),
        rag_sources=['neda_2024_2026', 'YahooFinanceForex'],
    ),
    Agent(
        name='Supply Chain Analyst',
        role='Refinery margins, logistics, importation costs',
        system_prompt=(
            'You are a supply chain analyst focused on Philippine petroleum importation. '
            'Analyze refinery margins, tanker freight rates, and terminal handling costs '
            'to estimate the landed cost component of the price change. '
            'End with: ' + ESTIMATE_LINE['gas']
        ),
        rag_sources=['YahooFinanceCrude', 'ManilaBulletin'],
    ),
    Agent(
        name='Consumer Analyst',
        role='Demand elasticity, commuter patterns, substitution effects',
        system_prompt=(
            'You are a consumer behavior expert analyzing how Filipino commuters '
            'and logistics operators respond to fuel price changes. Assess demand '
            'elasticity, jeepney modernization impact, and e-vehicle substitution. '
            'End with: ' + ESTIMATE_LINE['gas']
        ),
        rag_sources=['ManilaBulletin', 'neda_2024_2026'],
    ),
    Agent(
        name='Energy Policy Analyst',
        role='DOE pricing mechanism, auto-pricing formula, fuel subsidies',
        system_prompt=(
            'You are an energy policy analyst specializing in the Philippine DOE '
            'automatic oil pricing mechanism. Analyze how excise taxes, VAT, and '
            'any government subsidy buffer affect the consumer price change. '
            'End with: ' + ESTIMATE_LINE['gas']
        ),
        rag_sources=['neda_2024_2026', 'BusinessWorld'],
    ),
    Agent(
        name='Geopolitical Analyst',
        role='OPEC+ decisions, Middle East supply risks, shipping disruptions',
        system_prompt=(
            'You are a geopolitical analyst evaluating how OPEC+ production quotas, '
            'Middle East tensions, and Red Sea shipping disruptions affect crude '
            'supply available to Philippine refiners. '
            'End with: ' + ESTIMATE_LINE['gas']
        ),
        rag_sources=['YahooFinanceCrude', 'BusinessWorld'],
    ),
    Agent(
        name='Financial Strategist',
        role='Futures positioning, speculative premiums, hedging cost',
        system_prompt=(
            'You are a financial strategist analyzing crude oil futures markets. '
            'Evaluate speculative positioning, contango/backwardation structure, '
            'and hedging costs that oil companies pass on to Philippine consumers. '
            'End with: ' + ESTIMATE_LINE['gas']
        ),
        rag_sources=['YahooFinanceCrude', 'YahooFinanceForex'],
    ),
    Agent(
        name='Regional Economist',
        role='OFW remittances, provincial demand, inter-island freight',
        system_prompt=(
            'You are a regional development economist assessing how OFW remittance '
            'flows, provincial fuel demand patterns, and inter-island freight costs '
            'create price disparities across Philippine regions. '
            'End with: ' + ESTIMATE_LINE['gas']
        ),
        rag_sources=['neda_2024_2026', 'ManilaBulletin'],
    ),

    # ── 5 mini validator agents (fast tier) ───────────────────────────────────
    Agent(
        name='Data Validator',
        role='Sanity-checks estimates vs historical price change range',
        system_prompt=(
            'You are a data validator. Review all previous agent estimates and '
            'check whether the proposed price changes are within the historical '
            'range of Philippine fuel price adjustments (typically ±₱0.50–₱5.00/L). '
            'Flag any outlier. Give a brief validity verdict. '
            'End with: ' + ESTIMATE_LINE['gas']
        ),
        rag_sources=['YahooFinanceCrude', 'YahooFinanceForex'],
        tier=_MINI_TIER,
        is_mini=True,
    ),
    Agent(
        name='Sentiment Screener',
        role='News headline sentiment, public reaction, media framing',
        system_prompt=(
            'You are a sentiment screener. Scan the news context for market '
            'sentiment signals — are headlines bullish or bearish on fuel prices? '
            'How is media framing influencing public expectation? '
            'End with: ' + ESTIMATE_LINE['gas']
        ),
        rag_sources=['ManilaBulletin', 'BusinessWorld'],
        tier=_MINI_TIER,
        is_mini=True,
    ),
    Agent(
        name='Consensus Tracker',
        role='Monitors convergence across main agent estimates',
        system_prompt=(
            'You are a consensus tracker. Review all previous responses and determine '
            'whether the agents are converging or diverging. Identify the range, '
            'median, and any persistent disagreements. Offer a synthesized mid-point. '
            'End with: ' + ESTIMATE_LINE['gas']
        ),
        rag_sources=[],
        tier=_MINI_TIER,
        is_mini=True,
    ),
    Agent(
        name='Historical Comparator',
        role='Matches current scenario to past Philippine fuel price shocks',
        system_prompt=(
            'You are a historical comparator. Match the current scenario parameters '
            'to the closest historical episodes of Philippine fuel price changes '
            'from the NEDA data. What happened to prices then? Apply that lens. '
            'End with: ' + ESTIMATE_LINE['gas']
        ),
        rag_sources=['neda_2024_2026'],
        tier=_MINI_TIER,
        is_mini=True,
    ),
    Agent(
        name='Outlier Detector',
        role='Flags estimates diverging significantly from the group median',
        system_prompt=(
            'You are an outlier detector. Compare all agent price estimates. '
            'Identify which agent (if any) is significantly outside the group '
            'and explain why their reasoning may or may not be valid. '
            'Provide your own calibrated estimate. '
            'End with: ' + ESTIMATE_LINE['gas']
        ),
        rag_sources=[],
        tier=_MINI_TIER,
        is_mini=True,
    ),
]


_FOOD_ANCHOR = (
    f'IMPORTANT: Current Philippine food inflation is {_FOOD_INFLATION_YOY_PCT:.1f}% '
    f'year-on-year (PSA July 2026). Commercial rice (NCR) costs approximately '
    f'₱{_RICE_PRICE_PHP_KG:.0f}/kg. Typical monthly food price index changes in the '
    f'Philippines are ±0.3% to ±2.5%. Output only the signed monthly CHANGE percentage. '
)

FOOD_AGENTS: list[Agent] = [
    Agent(
        name='Agri Analyst',
        role='Crop supply, harvest cycles, import dependency',
        system_prompt=(
            _FOOD_ANCHOR +
            'You are an agricultural economist specializing in Philippine food markets. '
            'Using the gas price context and weather data provided, estimate the monthly '
            'food price index CHANGE. '
            'End your response with exactly one line: ' + ESTIMATE_LINE['food']
        ),
        rag_sources=['neda_2024_2026', 'WBPhilFood', 'NFARiceRetail'],
    ),
    Agent(
        name='Supply Chain Expert',
        role='Transport cost pass-through from fuel prices',
        system_prompt=(
            _FOOD_ANCHOR +
            'You are a logistics expert analyzing how fuel price changes cascade into '
            'Philippine food distribution costs. Using the gas price context, estimate '
            'transport cost contribution to food price change. '
            'End your response with exactly one line: ' + ESTIMATE_LINE['food']
        ),
        rag_sources=['YahooFinanceCrude', 'OpenMeteoManila'],
    ),
    Agent(
        name='Weather Interpreter',
        role='Rainfall and temperature effects on crop yields',
        system_prompt=(
            _FOOD_ANCHOR +
            'You are a climate-agriculture analyst. Using the rainfall and temperature '
            'data provided (weighted average across Central Luzon, Bicol, and Davao), '
            'assess crop stress and estimate weather-driven food price pressure. '
            'End your response with exactly one line: ' + ESTIMATE_LINE['food']
        ),
        rag_sources=['PAGASAWeather', 'OpenMeteoManila'],
    ),
    Agent(
        name='Trade Policy Critic',
        role='Tariff, NFA buffer stock, import quota impact',
        system_prompt=(
            _FOOD_ANCHOR +
            'You are a trade policy analyst focused on Philippine food security. '
            'Challenge or support previous estimates based on NFA buffer stocks, '
            'import quotas, and tariff policy. '
            'End your response with exactly one line: ' + ESTIMATE_LINE['food']
        ),
        rag_sources=['neda_2024_2026', 'WBPhilFood', 'NFARiceRetail'],
    ),
]

_ELEC_ANCHOR = (
    f'IMPORTANT: Current Meralco residential electricity rate is ₱{_MERALCO_RATE_PHP_KWH:.4f}/kWh '
    f'(July 2026). Typical monthly Meralco adjustments range from ±₱0.01 to ±₱1.50/kWh. '
    f'Output only the signed monthly CHANGE in ₱/kWh. '
)

ELECTRICITY_AGENTS: list[Agent] = [
    Agent(
        name='Energy Economist',
        role='Generation mix, fuel cost pass-through',
        system_prompt=(
            _ELEC_ANCHOR +
            'You are an energy economist specializing in Philippine power markets. '
            'Using the gas price context, estimate the monthly electricity rate change (PHP/kWh). '
            'End your response with exactly one line: ' + ESTIMATE_LINE['electricity']
        ),
        rag_sources=['YahooFinanceCrude', 'EIAElectricity', 'MeralcoCharge'],
    ),
    Agent(
        name='Grid Analyst',
        role='Meralco capacity, demand-supply balance',
        system_prompt=(
            _ELEC_ANCHOR +
            'You are a grid operations analyst for the Philippine electricity market. '
            'Assess demand-supply balance and its effect on Meralco distribution charges. '
            'End your response with exactly one line: ' + ESTIMATE_LINE['electricity']
        ),
        rag_sources=['EIAElectricity', 'OpenMeteoManila', 'WESMSpot'],
    ),
    Agent(
        name='Regulatory Expert',
        role='ERC rate review cycles, stranded cost recovery',
        system_prompt=(
            _ELEC_ANCHOR +
            'You are a regulatory affairs expert specializing in ERC proceedings. '
            'Analyze pending rate reviews and stranded cost recovery affecting the next billing period. '
            'End your response with exactly one line: ' + ESTIMATE_LINE['electricity']
        ),
        rag_sources=['neda_2024_2026', 'EIAElectricity', 'MeralcoCharge'],
    ),
    Agent(
        name='Demand Forecaster',
        role='Industrial and residential load outlook',
        system_prompt=(
            _ELEC_ANCHOR +
            'You are a demand forecasting analyst for Meralco service area. '
            'Estimate load growth and its effect on WESM spot prices. '
            'End your response with exactly one line: ' + ESTIMATE_LINE['electricity']
        ),
        rag_sources=['EIAElectricity', 'OpenMeteoManila', 'WESMSpot'],
    ),
]


_SYNTHESIZER_TIER = llm.DEEP


class SynthesizerThread(QThread):
    token_ready = pyqtSignal(str)
    finished    = pyqtSignal(str)

    def __init__(self, gas_verdict: str, food_verdict: str, elec_verdict: str, parent=None):
        super().__init__(parent)
        self._gas   = gas_verdict
        self._food  = food_verdict
        self._elec  = elec_verdict

    def run(self):
        messages = [
            {
                'role': 'system',
                'content': (
                    'You are a Philippine macroeconomic analyst synthesizing expert sector analysis. '
                    'Write 3-5 sentences summarizing the cascade effect across gas, food, and electricity. '
                    'Focus on the household impact. Be specific about direction and magnitude.'
                ),
            },
            {
                'role': 'user',
                'content': (
                    f'GAS SECTOR ANALYSIS:\n{self._gas}\n\n'
                    f'FOOD SECTOR ANALYSIS:\n{self._food}\n\n'
                    f'ELECTRICITY SECTOR ANALYSIS:\n{self._elec}\n\n'
                    'Provide a 3-5 sentence macro summary of the cascade effect on Philippine households.'
                ),
            },
        ]
        full_text = ''
        for token in llm.stream(messages, tier=_SYNTHESIZER_TIER):
            full_text += token
            self.token_ready.emit(token)
        self.finished.emit(full_text)


def _parse_think(text: str) -> tuple[str, str]:
    """Split <think>...</think> blocks from final statement. Returns (thinking, statement)."""
    # Capture content of all complete think blocks
    thinking_parts = re.findall(r'<think>(.*?)</think>', text, re.DOTALL)
    # Statement is the text after the last </think> tag
    last_close = text.rfind('</think>')
    if last_close != -1:
        statement = text[last_close + len('</think>'):]
    else:
        statement = text
    # Capture content of any unclosed <think> tag (truncated stream) into thinking
    unclosed = re.search(r'<think>(.*)', statement, flags=re.DOTALL)
    if unclosed:
        thinking_parts.append(unclosed.group(1))
        statement = re.sub(r'<think>.*', '', statement, flags=re.DOTALL)
    return ' '.join(thinking_parts).strip(), statement.strip()


# Parse-sanity bounds. These are not economic claims — they reject values that
# can only be a misparse. PSA month-on-month food CPI moves within a couple of
# percent, so a double-digit "monthly" figure is a year-on-year number that
# leaked out of the surrounding prose. Meralco's residential rate is ~₱14/kWh
# and moves by tenths of a peso month to month.
_MAX_REALISTIC_FOOD_PCT = 10.0
_MAX_REALISTIC_ELEC_PHP_KWH = 3.0
# Fuel moves by pesos, not tens of pesos, in a month. A "+150.00/L" is the model
# quoting an absolute pump PRICE where a CHANGE was asked for — the single most
# common misparse in live runs. This bound lived only in swarm.py, so the Forum
# (which uses `_extract_price` directly) had no guard at all and put +150.00/L on
# an agent card. It belongs here with the other two parse-sanity bounds.
_MAX_REALISTIC_FUEL_PHP_L = 8.0


# Tolerance-band phrases are instructions, not answers. "stay within +/-P1.00/L"
# echoed from the calibration rule contains a signed peso amount, and the prose
# fallback scraped the "-P1.00" out of it as an estimate. Stripped before any
# matching so a band can never masquerade as a change.
_TOLERANCE_BAND_RE = re.compile(
    r'(?:±|\+/-|\+-|within\s*[+\-±/]*)\s*(?:₱|PHP|P)?\s*\d+(?:\.\d+)?'
    r'(?:\s*/?\s*(?:L|kWh))?',
    re.IGNORECASE)

# Direction cues, used only to veto a contradictory unsigned parse.
_DECREASE_CUES = re.compile(
    r'\b(?:decreas|declin|drop|fall|fell|lower|down(?:ward)?|eas(?:e|es|ing)|reduc)',
    re.IGNORECASE)
_INCREASE_CUES = re.compile(
    r'\b(?:increas|ris(?:e|es|ing)|rose|climb|up(?:ward|tick)?|hik|surg)',
    re.IGNORECASE)


def _last_estimate_match(text: str, unit_pattern: str) -> Optional[float]:
    """Read the value off the agent's final `ESTIMATE:` line.

    Every agent prompt ends with an explicit "End with: ESTIMATE: ..."
    instruction, so that line — not the surrounding reasoning — is the answer.
    Taking the last match matters because agents restate and revise; taking the
    first would lock in a number the agent went on to reject.

    The sign is OPTIONAL here and defaults to positive. Callers still require an
    explicit sign in their *prose* fallback, where it is what distinguishes a
    change from a baseline mention like "a base of ₱60/L" — but on the ESTIMATE
    line the agent has been asked for a change, so an unsigned number is a change
    with the "+" dropped. Requiring it here cost real coverage: in a live 17-agent
    run, 11 agents produced an ESTIMATE line that this regex rejected purely for a
    missing plus. Relaxing it raises the misparse risk in exchange, which is what
    the per-sector plausibility ceilings exist to absorb.
    """
    text = _TOLERANCE_BAND_RE.sub(' ', text)
    hits = re.findall(
        rf'ESTIMATE\s*:\s*\**\s*([+\-])?\s*{unit_pattern}',
        text,
        flags=re.IGNORECASE,
    )
    if not hits:
        return None
    sign, raw = hits[-1]
    if not sign:
        # Unsigned defaults to positive — but not when the agent's own words argue
        # a fall. A live run produced est=+1.0 from an agent among all-negatives:
        # an unsigned number stamped positive against the statement's direction
        # becomes the outlier that drags the centre and the agreement metric.
        # Refusing (None) loses one data point; guessing poisons the consensus.
        if _DECREASE_CUES.search(text) and not _INCREASE_CUES.search(text):
            return None
    return (-1 if sign == '-' else 1) * float(raw)


def _extract_price(text: str) -> Optional[float]:
    """Extract a signed price change: +₱X.XX or -₱X.XX.

    Prefers the explicit ESTIMATE line; only falls back to scanning the prose
    when the agent did not produce one. Requires an explicit sign either way,
    so baseline mentions like 'a base of ₱60/L' are not mistaken for a change.

    Deliberately UNBOUNDED: `forum.py`'s `_extract_guarded` calls this directly
    and needs the raw value to report a rejected estimate as distinct from no
    estimate at all (see its own docstring). Callers that have no such
    accepted/rejected reporting layer of their own must apply
    `_MAX_REALISTIC_FUEL_PHP_L` themselves -- see `_extract_price_bounded`.
    """
    anchored = _last_estimate_match(text, r'(?:₱|PHP|P)?\s*(\d+\.?\d*)')
    if anchored is not None:
        return anchored
    m = re.search(r'([+\-])\s*₱(\d+\.?\d*)', _TOLERANCE_BAND_RE.sub(' ', text))
    if m:
        sign = -1 if m.group(1) == '-' else 1
        return sign * float(m.group(2))
    return None


def _extract_price_bounded(text: str) -> Optional[float]:
    """`_extract_price` with the parse-sanity ceiling applied.

    Classic-mode gas debate (`DebateEngine`'s default `price_extractor`) is the
    one live gas path with no separate accepted/rejected reporting layer --
    unlike `forum.py`'s `_extract_guarded`, which deliberately keeps
    `_extract_price` unbounded so it can report a rejected value distinctly
    from no estimate at all. Food and electricity already get this protection
    from their own extractors (`_extract_percent`, `_extract_electricity_change`);
    this closes the same hole for gas's default path, without touching
    `_extract_price` itself and breaking that other caller's reporting.
    """
    value = _extract_price(text)
    if value is None or abs(value) > _MAX_REALISTIC_FUEL_PHP_L:
        return None
    return value


def _extract_electricity_change(text: str) -> Optional[float]:
    """Electricity rate change in ₱/kWh, with a parse-sanity bound."""
    value = _extract_price(text)
    if value is None or abs(value) > _MAX_REALISTIC_ELEC_PHP_KWH:
        return None
    return value


def _extract_percent(text: str) -> Optional[float]:
    """Extract a signed percentage change: +X.X% or -X.X%.

    Anchored to the ESTIMATE line for a specific reason: agents routinely cite
    a year-on-year figure while reasoning ("food inflation ran 6.1% YoY, so I
    expect +0.4% this month"). A first-match scan of the prose returns the
    citation instead of the forecast — and that number then flows through the
    0.388 food basket weight straight into the projected-CPI headline.
    """
    value = _last_estimate_match(text, r'(\d+\.?\d*)\s*%')
    if value is None:
        m = re.search(r'([+\-])\s*(\d+\.?\d*)\s*%', text)
        if not m:
            return None
        value = (-1 if m.group(1) == '-' else 1) * float(m.group(2))
    if abs(value) > _MAX_REALISTIC_FOOD_PCT:
        return None
    return value


#: Food sub-category prompt labels, in the order they should appear in the
#: judge's closing instruction. Six PSA CPI sub-categories confirmed live
#: against openstat.psa.gov.ph during this feature's brainstorming (see
#: docs/superpowers/specs/2026-08-12-food-subcategory-forecast-design.md).
_CATEGORY_LABELS = {
    'rice': 'RICE', 'meat': 'MEAT', 'fish': 'FISH',
    'dairy_eggs': 'DAIRY_EGGS', 'vegetables': 'VEGETABLES', 'sugar': 'SUGAR',
}


def _extract_category_percents(text: str) -> dict[str, float]:
    """One signed-percent line per food sub-category (RICE:, MEAT:, ...).

    Deliberately a separate, self-contained parser rather than a refactor of
    `_last_estimate_match` (which every other extractor in this file depends
    on and is heavily tested) -- this reuses its two safe-to-share pieces,
    `_TOLERANCE_BAND_RE` and `_MAX_REALISTIC_FOOD_PCT`, without touching the
    well-tested ESTIMATE-line parser itself.

    A category whose line is missing, unparseable, or out of bound is simply
    absent from the returned dict -- never present as 0.0 or copied from
    another category's value. Takes the LAST match per category, same reason
    every other extractor in this file does: agents restate and revise.

    The sign is REQUIRED (unlike `_last_estimate_match`'s ESTIMATE line, where
    an unsigned number defaults to positive). Every one of the twelve worked
    examples in `_FOOD_CATEGORY_LINES` carries an explicit sign, so an unsigned
    match here is not a category read with the "+" dropped -- it is usually a
    year-on-year figure cited with no sign context ("Rice: 8.9% year-on-year")
    or the sign living in nearby prose the judge's own words already contradict
    ("Rice prices are falling. RICE: 0.4%"). Guessing a sign for either would
    fabricate a month-on-month reading; refusing (treating it as unparseable)
    is the honest answer.
    """
    cleaned = _TOLERANCE_BAND_RE.sub(' ', text)
    result: dict[str, float] = {}
    for category, label in _CATEGORY_LABELS.items():
        hits = re.findall(
            rf'\b{label}\s*:\s*\**\s*([+\-])\s*(\d+\.?\d*)\s*%',
            cleaned, flags=re.IGNORECASE,
        )
        if not hits:
            continue
        sign, raw = hits[-1]
        value = (-1 if sign == '-' else 1) * float(raw)
        if abs(value) <= _MAX_REALISTIC_FOOD_PCT:
            result[category] = value
    return result


_CATEGORY_LINE_RE = re.compile(
    r'\b(?:' + '|'.join(re.escape(lbl) for lbl in _CATEGORY_LABELS.values()) +
    r')\s*:\s*\**\s*[+\-]?\s*\d+\.?\d*\s*%[^\n]*',
    re.IGNORECASE,
)


def _strip_category_lines(text: str) -> str:
    """Remove every RICE:/MEAT:/.../SUGAR: line -- label, value, AND any
    trailing commentary on that same line -- from text, so the blended
    ESTIMATE:'s prose fallback (which grabs the first signed percent it sees
    anywhere) can't accidentally pick up a category value, or a stray signed
    percent in that line's own trailing commentary, instead. Built
    generically from `_CATEGORY_LABELS` rather than a separate hardcoded list,
    so it can't drift out of sync with the six labels.

    Why this exists: `_extract_percent`'s prose fallback (a regex matching a
    leading plus-or-minus sign followed by a number and a percent sign)
    matches the FIRST signed percent anywhere in the text whenever the
    anchored ESTIMATE: line fails to parse (e.g. the judge writes
    "ESTIMATE: broadly unchanged"). The food judge prompt appends the six
    category lines AFTER its ESTIMATE line -- the judge is asked for the
    blended ESTIMATE first, then the six categories -- so that fallback
    would silently grab a category value (typically RICE, the first category
    line) and it would become the app's headline "Food" estimate with no
    signal anything went wrong. Stripping the category lines -- the whole
    line, not just the label+value token, since trailing commentary can
    itself contain a signed percent -- before the blended extraction closes
    that hole without touching the well-tested
    `_extract_percent`/`_last_estimate_match` themselves.
    """
    return _CATEGORY_LINE_RE.sub(' ', text)


#: Per-sector agreement band for `consensus()`: PHP/L for gas, percentage
#: points for food, PHP/kWh for electricity. Must track `forum._BAND` and (for
#: gas) `swarm._AGREEMENT_BAND` -- three places measuring the same three
#: sectors' agreement, and a room that reads as agreeing on one screen should
#: read the same way on another. This engine used one hardcoded 0.20 for every
#: sector: less than half gas's validated 0.50 (classic-mode gas understated
#: its own room's agreement relative to swarm mode) and double electricity's
#: validated 0.10 (classic-mode electricity overstated it).
_CONSENSUS_BAND = {'gas': 0.50, 'food': 0.3, 'electricity': 0.10}


class DebateEngine:
    def __init__(self, agents: list[Agent], rag: RagEngine, scenario: dict,
                 price_extractor=None, data_brief: Optional['LiveDataBrief'] = None,
                 anchor_note: str = '', sector: str = 'gas'):
        """
        scenario keys: oil_pct, usd_pct, bsp_rate, demand_index
        price_extractor: callable(text) -> Optional[float]. Defaults to _extract_price.
                         Pass _extract_percent for food agents.
        data_brief: LiveDataBrief instance injected into every agent prompt.
        anchor_note: optional physical/statistical baseline injected into every
                     prompt so a weak model reasons from the right scale rather
                     than inventing one — the sector equivalent of the master
                     judge's mechanical pass-through line.
        sector: names this engine in the sampling seed. Food and electricity run
                two DebateEngines in one process against the same scenario, so
                without the name they would request identical seeds and the two
                sectors would answer in unison.
        """
        self._agents = agents
        self._rag = rag
        self._scenario = scenario
        self._history: list[AgentResponse] = []
        self._price_extractor = price_extractor or _extract_price_bounded
        self._data_brief = data_brief
        self._anchor_note = anchor_note
        self._sector = sector

    def _seed(self, *parts: object) -> int:
        """Sampling seed for one call in this debate.

        This path had no seed at all: every `llm.stream` call in the file omitted
        it, so Ollama drew a fresh random seed per call at temperature 0.2. The
        food and electricity numbers were unreproducible even when the gas number
        stored on the same run row was seeded. ADR-002 covered the swarm and the
        Forum and missed the two sectors that share their run.

        Keyed on the vintage, like the swarm and the Forum, so it holds still for
        as long as the pricing week does. See `swarm._vintage_seed` for why the
        scenario itself is deliberately not in the key.
        """
        from ph_economic_ai.engine import vintage
        v = vintage.vintage()
        return llm.derive_seed(v['fuel_cycle'], v['day'], self._sector, *parts)

    def _scenario_text(self) -> str:
        s = self._scenario
        price = s.get('current_price', _GAS_PRICE_PHP_L)
        return (
            f"Current NCR retail gasoline: ₱{price:.2f}/L. "
            f"Scenario: oil price {s.get('oil_pct', 0):+.1f}%, "
            f"USD/PHP {s.get('usd_pct', 0):+.1f}%, "
            f"BSP rate {s.get('bsp_rate', 6.5):.2f}%, "
            f"demand index {s.get('demand_index', 72):.0f}."
        )

    def _build_prompt(self, agent: Agent, round_num: int) -> list[dict]:
        scenario_text = self._scenario_text()
        chunks = self._rag.query(scenario_text, top_k=5, sources=agent.rag_sources)
        rag_text = '\n'.join(
            f"[{c['source']}] {c['text'][:300]}" for c in chunks
        ) or 'No context retrieved.'
        prior = '\n'.join(
            f"{r.agent_name} (Round {r.round_num}): {r.statement[:400]}"
            for r in self._history
        )
        # Prepend live data brief when available — agents cite real numbers
        brief_block = ''
        if self._data_brief is not None:
            try:
                brief_block = self._data_brief.as_prompt_block(self._scenario) + '\n\n'
            except Exception:
                pass

        anchor_block = f"{self._anchor_note}\n\n" if self._anchor_note else ''
        user_content = (
            f"{brief_block}"
            f"{scenario_text}\n\n"
            f"{anchor_block}"
            f"Relevant context:\n{rag_text}\n\n"
            + (f"Previous agent responses:\n{prior}\n\n" if prior else '')
            # Instructions, not templates — see `unfilled_scaffold` and the note
            # at `forum._EST_LINE`. A small model copies a bracketed template
            # verbatim, which parses to nothing and puts placeholder words on the
            # card as if they were findings.
            + "Give your analysis. You MUST cite specific data from the DATA BRIEF "
            "when available. End your response with BOTH of these lines:\n"
            "CAUSAL CHAIN: <name the trigger> → <the market it moves> → "
            "<how that reaches the price> → <what a household pays> "
            "(worked example — \"CAUSAL CHAIN: Brent +5% → import cost up → "
            "weekly pass-through → jeepney fares rise\". Write your own chain; "
            "never copy the words in the angle brackets.)\n"
            "ESTIMATE: <the peso-per-litre CHANGE you expect, signed> "
            "(worked example — \"ESTIMATE: +0.85/L\" or \"ESTIMATE: -0.40/L\". "
            "Write your own number; never write X.XX.)"
        )
        return [
            {'role': 'system', 'content': agent.system_prompt},
            {'role': 'user', 'content': user_content},
        ]

    def run(
        self,
        rounds: int = 3,
        on_token: Optional[Callable[[str, str], None]] = None,
        on_agent_done: Optional[Callable[['AgentResponse'], None]] = None,
    ) -> list[AgentResponse]:
        """Run debate. on_token(agent_name, token), on_agent_done(response)."""
        self._history.clear()
        for round_num in range(1, rounds + 1):
            for agent in self._agents:
                messages = self._build_prompt(agent, round_num)
                full_text = ''
                for token in llm.stream(messages, tier=agent.tier, max_tokens=750,
                                        seed=self._seed(agent.name, round_num)):
                    full_text += token
                    if on_token:
                        on_token(agent.name, token)
                thinking, statement = _parse_think(full_text)
                response = AgentResponse(
                    agent_name=agent.name,
                    round_num=round_num,
                    thinking=thinking,
                    statement=statement,
                    price_estimate=self._price_extractor(statement),
                )
                self._history.append(response)
                if on_agent_done:
                    on_agent_done(response)
        return self._history

    def ask(
        self,
        agent_name: str,
        question: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Single follow-up call to one agent using full debate context."""
        agent = next((a for a in self._agents if a.name == agent_name), None)
        if agent is None:
            return ''
        prior = '\n'.join(
            f"{r.agent_name} (Round {r.round_num}): {r.statement[:300]}"
            for r in self._history
        )
        messages = [
            {'role': 'system', 'content': agent.system_prompt},
            {'role': 'user', 'content': (
                f"Debate history:\n{prior}\n\nFollow-up: {question}"
            )},
        ]
        full_text = ''
        # The question is part of the seed: two different follow-ups to one agent
        # are two different calls, and seeding only on the agent would make the
        # second answer a resample of the first.
        for token in llm.stream(messages, tier=agent.tier,
                                seed=self._seed('ask', agent_name, question)):
            full_text += token
            if on_token:
                on_token(token)
        _, statement = _parse_think(full_text)
        return statement

    def consensus(self) -> dict:
        """Compute final round consensus from history. Returns summary dict."""
        final_round = max((r.round_num for r in self._history), default=0)
        final = [r for r in self._history if r.round_num == final_round]
        estimates = [r.price_estimate for r in final if r.price_estimate is not None]
        if not estimates:
            return {'weighted_avg': None, 'low': None, 'high': None,
                    'confidence_pct': 0, 'verdicts': []}
        avg = sum(estimates) / len(estimates)
        band = _CONSENSUS_BAND.get(self._sector, 0.20)
        within = sum(1 for e in estimates if abs(e - avg) <= band)
        return {
            'weighted_avg': avg,
            'low': min(estimates),
            'high': max(estimates),
            'confidence_pct': int(within / len(estimates) * 100),
            'verdicts': [
                {'agent': r.agent_name, 'estimate': r.price_estimate,
                 'statement': r.statement}
                for r in final
            ],
        }


class DebateThread(QThread):
    """Runs DebateEngine.run() off the main thread; emits signals per token/agent."""
    token_received = pyqtSignal(str, str)          # agent_name, token
    agent_done = pyqtSignal(object)                 # AgentResponse
    debate_complete = pyqtSignal(object)            # list[AgentResponse]
    error_occurred = pyqtSignal(str)

    def __init__(self, engine: DebateEngine, rounds: int = 3, parent=None):
        super().__init__(parent)
        self._engine = engine
        self._rounds = rounds

    def run(self):
        try:
            responses = self._engine.run(
                rounds=self._rounds,
                on_token=lambda name, tok: self.token_received.emit(name, tok),
                on_agent_done=lambda r: self.agent_done.emit(r),
            )
            self.debate_complete.emit(responses)
        except Exception as e:
            self.error_occurred.emit(f"{type(e).__name__}: {e}")
