import re
from dataclasses import dataclass
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
_FOOD_INFLATION_YOY_PCT: float = 6.1      # PSA April 2026, year-on-year
_RICE_PRICE_PHP_KG: float = 52.0          # Commercial rice, NCR avg
_MERALCO_RATE_PHP_KWH: float = 14.3345   # Meralco residential May 2026
_GAS_PRICE_PHP_L: float = 98.82           # NCR Unleaded 91 avg May 20 2026


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
            'ESTIMATE: +₱X.XX/L or ESTIMATE: -₱X.XX/L'
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
            'ESTIMATE: +₱X.XX/L or ESTIMATE: -₱X.XX/L'
        ),
        rag_sources=['neda_2024_2026', 'YahooFinanceForex'],
    ),
    Agent(
        name='Risk Assessor',
        role='Tail risks, remittances, demand shocks, supply gaps',
        system_prompt=(
            'You are a risk assessor. Identify tail risks and softening factors '
            'the other agents may have missed. End your response with exactly one line: '
            'ESTIMATE: +₱X.XX/L or ESTIMATE: -₱X.XX/L'
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
            'End with: ESTIMATE: +₱X.XX/L or ESTIMATE: -₱X.XX/L'
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
            'End with: ESTIMATE: +₱X.XX/L or ESTIMATE: -₱X.XX/L'
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
            'End with: ESTIMATE: +₱X.XX/L or ESTIMATE: -₱X.XX/L'
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
            'End with: ESTIMATE: +₱X.XX/L or ESTIMATE: -₱X.XX/L'
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
            'End with: ESTIMATE: +₱X.XX/L or ESTIMATE: -₱X.XX/L'
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
            'End with: ESTIMATE: +₱X.XX/L or ESTIMATE: -₱X.XX/L'
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
            'End with: ESTIMATE: +₱X.XX/L or ESTIMATE: -₱X.XX/L'
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
            'End with: ESTIMATE: +₱X.XX/L or ESTIMATE: -₱X.XX/L'
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
            'End with: ESTIMATE: +₱X.XX/L or ESTIMATE: -₱X.XX/L'
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
            'End with: ESTIMATE: +₱X.XX/L or ESTIMATE: -₱X.XX/L'
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
            'End with: ESTIMATE: +₱X.XX/L or ESTIMATE: -₱X.XX/L'
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
            'End with: ESTIMATE: +₱X.XX/L or ESTIMATE: -₱X.XX/L'
        ),
        rag_sources=[],
        tier=_MINI_TIER,
        is_mini=True,
    ),
]


_FOOD_ANCHOR = (
    f'IMPORTANT: Current Philippine food inflation is {_FOOD_INFLATION_YOY_PCT:.1f}% '
    f'year-on-year (PSA April 2026). Commercial rice (NCR) costs approximately '
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
            'End your response with exactly one line: ESTIMATE: +X.X% or ESTIMATE: -X.X%'
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
            'End your response with exactly one line: ESTIMATE: +X.X% or ESTIMATE: -X.X%'
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
            'End your response with exactly one line: ESTIMATE: +X.X% or ESTIMATE: -X.X%'
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
            'End your response with exactly one line: ESTIMATE: +X.X% or ESTIMATE: -X.X%'
        ),
        rag_sources=['neda_2024_2026', 'WBPhilFood', 'NFARiceRetail'],
    ),
]

_ELEC_ANCHOR = (
    f'IMPORTANT: Current Meralco residential electricity rate is ₱{_MERALCO_RATE_PHP_KWH:.4f}/kWh '
    f'(May 2026). Typical monthly Meralco adjustments range from ±₱0.01 to ±₱1.50/kWh. '
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
            'End your response with exactly one line: ESTIMATE: +₱X.XX/kWh or ESTIMATE: -₱X.XX/kWh'
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
            'End your response with exactly one line: ESTIMATE: +₱X.XX/kWh or ESTIMATE: -₱X.XX/kWh'
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
            'End your response with exactly one line: ESTIMATE: +₱X.XX/kWh or ESTIMATE: -₱X.XX/kWh'
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
            'End your response with exactly one line: ESTIMATE: +₱X.XX/kWh or ESTIMATE: -₱X.XX/kWh'
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


def _extract_price(text: str) -> Optional[float]:
    """Extract signed price change: +₱X.XX or -₱X.XX. Requires explicit sign to avoid
    matching baseline price mentions like 'a base of ₱60/L'."""
    m = re.search(r'([+\-])\s*₱(\d+\.?\d*)', text)
    if m:
        sign = -1 if m.group(1) == '-' else 1
        return sign * float(m.group(2))
    return None


def _extract_percent(text: str) -> Optional[float]:
    """Extract signed percentage change: +X.X% or -X.X% (for food index estimates)."""
    m = re.search(r'([+\-])\s*(\d+\.?\d*)\s*%', text)
    if m:
        sign = -1 if m.group(1) == '-' else 1
        return sign * float(m.group(2))
    return None


class DebateEngine:
    def __init__(self, agents: list[Agent], rag: RagEngine, scenario: dict,
                 price_extractor=None, data_brief: Optional['LiveDataBrief'] = None):
        """
        scenario keys: oil_pct, usd_pct, bsp_rate, demand_index
        price_extractor: callable(text) -> Optional[float]. Defaults to _extract_price.
                         Pass _extract_percent for food agents.
        data_brief: LiveDataBrief instance injected into every agent prompt.
        """
        self._agents = agents
        self._rag = rag
        self._scenario = scenario
        self._history: list[AgentResponse] = []
        self._price_extractor = price_extractor or _extract_price
        self._data_brief = data_brief

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

        user_content = (
            f"{brief_block}"
            f"{scenario_text}\n\n"
            f"Relevant context:\n{rag_text}\n\n"
            + (f"Previous agent responses:\n{prior}\n\n" if prior else '')
            + "Give your analysis. You MUST cite specific data from the DATA BRIEF "
            "when available. End your response with BOTH of these lines:\n"
            "CAUSAL CHAIN: [trigger] → [market effect] → [price mechanism] → [consumer impact]\n"
            "ESTIMATE: +₱X.XX/L or ESTIMATE: -₱X.XX/L"
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
                for token in llm.stream(messages, tier=agent.tier, max_tokens=750):
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
        for token in llm.stream(messages, tier=agent.tier):
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
        within = sum(1 for e in estimates if abs(e - avg) <= 0.20)
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
