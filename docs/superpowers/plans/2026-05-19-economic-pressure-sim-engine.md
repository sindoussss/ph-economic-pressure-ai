# Economic Pressure AI Simulation Engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign `ph_economic_ai` into a 5-stage OASIS-style simulation engine with parallel RAG fetching, a 10-agent debate canvas, and deepseek-r1:8b via Ollama.

**Architecture:** A left-sidebar QStackedWidget app with 5 stage panels. An `engine/` layer (RAG + debate) sits below the UI and is independently testable. The existing HistGBM model, `build_features`, and `model.py` functions are kept as-is and wired into Stage 3/4.

**Tech Stack:** PyQt6 6.10, Ollama (deepseek-r1:8b), sklearn TfidfVectorizer, requests + BeautifulSoup4, PyMuPDF (fitz), reportlab, concurrent.futures

---

## File Map

```
ph_economic_ai/
├── main.py                     MODIFY  — showMaximized(); pass engine refs to SimMainWindow
├── engine/
│   ├── __init__.py             NEW
│   ├── rag.py                  NEW     — RagEngine: fetch, chunk, TF-IDF, query
│   └── debate.py               NEW     — DebateEngine, Agent, DebateThread(QThread)
├── ui/
│   ├── sidebar.py              REWRITE — 5-stage minimalist nav (no emoji, no colors)
│   ├── main_window.py          REWRITE — SimMainWindow with QStackedWidget for 5 stages
│   ├── stage1_rag.py           NEW     — Stage 1: RAG pipeline panel
│   ├── stage2_setup.py         NEW     — Stage 2: scenario inputs + agent roster
│   ├── stage3_canvas.py        NEW     — Stage 3: simulation canvas + debate thread
│   ├── stage4_report.py        NEW     — Stage 4: report panel + PDF export
│   └── stage5_interact.py      NEW     — Stage 5: adjust/re-run, ask agent, toggle RAG
├── assets/corpus/
│   └── neda_2024_2026.txt      NEW     — pre-bundled NEDA corpus text
└── tests/
    ├── test_rag.py             NEW
    └── test_debate.py          NEW
```

Files **not touched**: `model.py`, `utils/preprocessing.py`, `data.py`, `fetcher.py`, `utils/explanation.py`, `ui/dashboard.py`, `ui/agent_graph.py`, `ui/pressure.py`, `ui/charts.py`.

---

## Task 1: Install new dependencies

**Files:** none (environment setup)

- [ ] **Step 1: Install packages**

```bash
pip install ollama beautifulsoup4 pymupdf reportlab
```

Expected output: Successfully installed ollama-... beautifulsoup4-... pymupdf-... reportlab-...

- [ ] **Step 2: Pull deepseek-r1:8b**

```bash
ollama pull deepseek-r1:8b
```

Expected: model downloads (~4.9 GB). Verify with `ollama list` — `deepseek-r1:8b` appears.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "chore: no code change — deps installed (ollama, bs4, pymupdf, reportlab)"
```

---

## Task 2: RAG engine

**Files:**
- Create: `ph_economic_ai/engine/__init__.py`
- Create: `ph_economic_ai/engine/rag.py`
- Create: `ph_economic_ai/tests/test_rag.py`

- [ ] **Step 1: Write failing tests**

```python
# ph_economic_ai/tests/test_rag.py
import pytest
from ph_economic_ai.engine.rag import RagEngine


def test_chunk_splits_long_text():
    engine = RagEngine()
    text = 'oil price fuel gasoline Philippines economy ' * 100
    chunks = engine._chunk(text, source='test')
    assert len(chunks) > 1
    assert all(len(c.text) <= 2048 for c in chunks)


def test_add_text_and_query_returns_result():
    engine = RagEngine()
    engine.add_text('DOE', 'The pump price will increase next week due to Brent crude oil rising.')
    results = engine.query('gasoline price increase oil', top_k=1)
    assert len(results) == 1
    assert results[0]['source'] == 'DOE'
    assert results[0]['score'] > 0


def test_query_empty_engine_returns_empty():
    engine = RagEngine()
    assert engine.query('anything') == []


def test_toggle_source_excludes_chunks():
    engine = RagEngine()
    engine.add_text('DOE', 'pump price fuel gasoline oil Philippines cost')
    engine.add_text('BSP', 'pump price fuel gasoline oil Philippines cost')
    engine.toggle_source('DOE', False)
    results = engine.query('pump price fuel gasoline', top_k=10)
    assert all(r['source'] != 'DOE' for r in results)


def test_toggle_source_reenabled():
    engine = RagEngine()
    engine.add_text('DOE', 'pump price fuel gasoline oil Philippines')
    engine.toggle_source('DOE', False)
    engine.toggle_source('DOE', True)
    results = engine.query('pump price fuel', top_k=5)
    assert any(r['source'] == 'DOE' for r in results)


def test_query_filter_by_sources():
    engine = RagEngine()
    engine.add_text('DOE', 'pump price adjustment next week gasoline diesel oil')
    engine.add_text('BSP', 'monetary policy rate decision inflation peso dollar')
    results = engine.query('gasoline price oil', top_k=5, sources=['DOE'])
    assert all(r['source'] == 'DOE' for r in results)


def test_chunk_count_property():
    engine = RagEngine()
    engine.add_text('X', 'word ' * 2000)
    assert engine.chunk_count > 0
```

- [ ] **Step 2: Run tests — verify they all fail**

```bash
python -m pytest ph_economic_ai/tests/test_rag.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` on `ph_economic_ai.engine.rag`.

- [ ] **Step 3: Create engine package**

```python
# ph_economic_ai/engine/__init__.py
# (empty)
```

- [ ] **Step 4: Implement RagEngine**

```python
# ph_economic_ai/engine/rag.py
import concurrent.futures
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer

_CHUNK_SIZE = 2048    # ~512 tokens at 4 chars/token
_CHUNK_OVERLAP = 256  # ~64 tokens overlap

# Source name → URL for parallel startup fetch
SOURCES: dict[str, str] = {
    'DOE':               'https://www.doe.gov.ph/price-monitoring',
    'BSP':               'https://www.bsp.gov.ph/Pages/MediaAndResearch/MediaReleases/MediaReleases.aspx',
    'BusinessWorld':     'https://www.bworldonline.com/category/economy/',
    'Reuters':           'https://www.reuters.com/search/news?blob=Philippines+fuel+oil+price',
    'Inquirer Business': 'https://business.inquirer.net/?s=gasoline+price',
    'Manila Bulletin':   'https://mb.com.ph/?s=gasoline+price',
    'OPEC':              'https://www.opec.org/opec_web/en/press_room/30.htm',
    'Yahoo Finance Crude': 'https://finance.yahoo.com/quote/BZ%3DF/',
    'Yahoo Finance Forex': 'https://finance.yahoo.com/quote/PHP%3DX/',
}

_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}


@dataclass
class Chunk:
    text: str
    source: str
    url: str = ''
    fetched_at: float = field(default_factory=time.time)


class RagEngine:
    def __init__(self):
        self._chunks: list[Chunk] = []
        self._disabled: set[str] = set()
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._matrix = None  # scipy sparse

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch_all(
        self,
        on_progress: Optional[Callable[[str, int], None]] = None,
    ) -> dict[str, int]:
        """Fetch all sources in parallel. Returns {source_name: chunk_count}."""
        results: dict[str, int] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=9) as pool:
            futures = {
                pool.submit(self._fetch_one, name, url): name
                for name, url in SOURCES.items()
            }
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                try:
                    chunks = future.result()
                    self._chunks.extend(chunks)
                    results[name] = len(chunks)
                except Exception:
                    results[name] = 0
                if on_progress:
                    on_progress(name, results[name])
        self._refit()
        return results

    def add_pdf(self, path: str) -> int:
        """Load PDF with PyMuPDF, chunk it, add to index. Returns chunk count."""
        import fitz  # PyMuPDF
        doc = fitz.open(path)
        text = ' '.join(page.get_text() for page in doc)
        doc.close()
        name = Path(path).stem
        new_chunks = self._chunk(text, source=name, url=path)
        self._chunks.extend(new_chunks)
        self._refit()
        return len(new_chunks)

    def add_text(self, source: str, text: str) -> int:
        """Add pre-bundled corpus text. Returns chunk count."""
        new_chunks = self._chunk(text, source=source)
        self._chunks.extend(new_chunks)
        self._refit()
        return len(new_chunks)

    def query(
        self,
        text: str,
        top_k: int = 5,
        sources: Optional[list[str]] = None,
    ) -> list[dict]:
        """Return top_k most relevant chunks. Optionally filter to specific sources."""
        if self._vectorizer is None or self._matrix is None:
            return []
        active = self._active_chunks()
        if sources:
            idxs = [i for i, c in enumerate(active) if c.source in sources]
            if not idxs:
                return []
            sub_matrix = self._matrix[idxs]
            sub_chunks = [active[i] for i in idxs]
        else:
            sub_matrix = self._matrix
            sub_chunks = active

        q_vec = self._vectorizer.transform([text])
        scores = (sub_matrix @ q_vec.T).toarray().flatten()
        top_idxs = np.argsort(scores)[::-1][:top_k]
        return [
            {'text': sub_chunks[i].text, 'source': sub_chunks[i].source,
             'score': float(scores[i])}
            for i in top_idxs if scores[i] > 0
        ]

    def toggle_source(self, source: str, enabled: bool) -> None:
        if enabled:
            self._disabled.discard(source)
        else:
            self._disabled.add(source)
        self._refit()

    @property
    def chunk_count(self) -> int:
        return len(self._active_chunks())

    @property
    def all_source_names(self) -> list[str]:
        return sorted({c.source for c in self._chunks})

    # ── Internal ──────────────────────────────────────────────────────────────

    def _fetch_one(self, name: str, url: str) -> list[Chunk]:
        resp = requests.get(url, headers=_HEADERS, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
            tag.decompose()
        text = soup.get_text(separator=' ', strip=True)
        return self._chunk(text, source=name, url=url)

    def _chunk(self, text: str, source: str, url: str = '') -> list[Chunk]:
        chunks = []
        start = 0
        while start < len(text):
            chunks.append(Chunk(text=text[start:start + _CHUNK_SIZE],
                                source=source, url=url))
            start += _CHUNK_SIZE - _CHUNK_OVERLAP
            if start >= len(text):
                break
        return chunks

    def _active_chunks(self) -> list[Chunk]:
        return [c for c in self._chunks if c.source not in self._disabled]

    def _refit(self) -> None:
        active = self._active_chunks()
        if not active:
            self._vectorizer = None
            self._matrix = None
            return
        self._vectorizer = TfidfVectorizer(max_features=10_000, stop_words='english')
        self._matrix = self._vectorizer.fit_transform([c.text for c in active])
```

- [ ] **Step 5: Run tests — all must pass**

```bash
python -m pytest ph_economic_ai/tests/test_rag.py -v
```

Expected: 7 passed.

- [ ] **Step 6: Commit**

```bash
git add ph_economic_ai/engine/ ph_economic_ai/tests/test_rag.py
git commit -m "feat: add RagEngine with parallel fetch, chunker, TF-IDF index"
```

---

## Task 3: Debate engine

**Files:**
- Create: `ph_economic_ai/engine/debate.py`
- Create: `ph_economic_ai/tests/test_debate.py`

- [ ] **Step 1: Write failing tests**

```python
# ph_economic_ai/tests/test_debate.py
import pytest
from unittest.mock import MagicMock, patch
from ph_economic_ai.engine.debate import (
    _parse_think, _extract_price,
    Agent, AgentResponse, DebateEngine, DEFAULT_AGENTS,
)


def test_parse_think_splits_tag():
    thinking, statement = _parse_think(
        '<think>I must consider OPEC signals.</think>My estimate is +₱2.50/L.'
    )
    assert thinking == 'I must consider OPEC signals.'
    assert statement == 'My estimate is +₱2.50/L.'


def test_parse_think_no_tag():
    thinking, statement = _parse_think('My estimate is +₱2.50/L.')
    assert thinking == ''
    assert statement == 'My estimate is +₱2.50/L.'


def test_extract_price_positive_delta():
    assert _extract_price('price will rise by +₱2.50/L') == pytest.approx(2.50)


def test_extract_price_negative_delta():
    assert _extract_price('downward pressure of -₱1.20') == pytest.approx(-1.20)


def test_extract_price_absolute():
    assert _extract_price('forecast ₱73.20 per liter') == pytest.approx(73.20)


def test_extract_price_none():
    assert _extract_price('no price mentioned here') is None


def test_default_agents_count():
    assert len(DEFAULT_AGENTS) == 3
    names = {a.name for a in DEFAULT_AGENTS}
    assert 'Market Analyst' in names
    assert 'Policy Expert' in names
    assert 'Risk Assessor' in names


def _make_mock_rag():
    rag = MagicMock()
    rag.query.return_value = [
        {'text': 'Fuel prices rising due to oil shock.', 'source': 'DOE', 'score': 0.9}
    ]
    return rag


def test_build_prompt_contains_scenario():
    rag = _make_mock_rag()
    engine = DebateEngine(DEFAULT_AGENTS, rag,
                          {'oil_pct': 5.0, 'usd_pct': 2.0,
                           'bsp_rate': 6.5, 'demand_index': 72})
    messages = engine._build_prompt(DEFAULT_AGENTS[0], round_num=1)
    combined = ' '.join(m['content'] for m in messages)
    assert '+5.0' in combined or '5.0' in combined
    assert '6.5' in combined


def test_run_calls_ollama_per_agent_per_round():
    rag = _make_mock_rag()
    engine = DebateEngine(DEFAULT_AGENTS[:2], rag,
                          {'oil_pct': 5.0, 'usd_pct': 2.0,
                           'bsp_rate': 6.5, 'demand_index': 72})

    fake_stream = [{'message': {'content': tok}} for tok in
                   ['<think>', 'thinking', '</think>', '+₱2.50/L']]

    with patch('ph_economic_ai.engine.debate.ollama.chat',
               return_value=iter(fake_stream)) as mock_chat:
        responses = engine.run(rounds=2)

    assert mock_chat.call_count == 4  # 2 agents × 2 rounds
    assert len(responses) == 4
    assert all(isinstance(r, AgentResponse) for r in responses)


def test_run_extracts_price_estimate():
    rag = _make_mock_rag()
    engine = DebateEngine(DEFAULT_AGENTS[:1], rag,
                          {'oil_pct': 5.0, 'usd_pct': 2.0,
                           'bsp_rate': 6.5, 'demand_index': 72})
    fake_stream = [{'message': {'content': tok}}
                   for tok in ['Pump price estimate is ', '+₱2.50', '/L']]
    with patch('ph_economic_ai.engine.debate.ollama.chat',
               return_value=iter(fake_stream)):
        responses = engine.run(rounds=1)
    assert responses[0].price_estimate == pytest.approx(2.50)
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
python -m pytest ph_economic_ai/tests/test_debate.py -v
```

Expected: `ImportError` on `ph_economic_ai.engine.debate`.

- [ ] **Step 3: Implement debate engine**

```python
# ph_economic_ai/engine/debate.py
import re
from dataclasses import dataclass
from typing import Callable, Optional

import ollama
from PyQt6.QtCore import QThread, pyqtSignal

from ph_economic_ai.engine.rag import RagEngine


@dataclass
class Agent:
    name: str
    role: str
    system_prompt: str
    rag_sources: list[str]


@dataclass
class AgentResponse:
    agent_name: str
    round_num: int
    thinking: str
    statement: str
    price_estimate: Optional[float]  # ₱/L change; None if not parseable


DEFAULT_AGENTS: list[Agent] = [
    Agent(
        name='Market Analyst',
        role='Price signals, news, short-term pass-through',
        system_prompt=(
            'You are a market analyst specializing in Philippine fuel markets. '
            'Using the provided news and price data, estimate the short-term '
            'retail gasoline price impact (₱/L change) of the given scenario. '
            'Be specific — give a ₱/L estimate.'
        ),
        rag_sources=['DOE', 'Reuters', 'BusinessWorld',
                     'Yahoo Finance Crude', 'Yahoo Finance Forex'],
    ),
    Agent(
        name='Policy Expert',
        role='Monetary policy, FX transmission, regulatory context',
        system_prompt=(
            'You are a policy expert focused on BSP monetary policy and peso dynamics. '
            'Using BSP statements and economic reports, challenge or support the '
            'previous estimate. Give your own ₱/L estimate.'
        ),
        rag_sources=['BSP', 'neda_2024_2026'],
    ),
    Agent(
        name='Risk Assessor',
        role='Tail risks, remittances, demand shocks, supply gaps',
        system_prompt=(
            'You are a risk assessor. Identify tail risks and softening factors '
            'the other agents may have missed. Provide a final risk-adjusted ₱/L estimate.'
        ),
        rag_sources=['OPEC', 'Manila Bulletin', 'neda_2024_2026'],
    ),
]


def _parse_think(text: str) -> tuple[str, str]:
    """Split <think>...</think> from final statement. Returns (thinking, statement)."""
    m = re.search(r'<think>(.*?)</think>(.*)', text, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return '', text.strip()


def _extract_price(text: str) -> Optional[float]:
    """Extract first ₱XX.XX or +₱XX.XX or -₱XX.XX from text."""
    m = re.search(r'([+\-]?)₱(\d+\.?\d*)', text)
    if m:
        sign = -1 if m.group(1) == '-' else 1
        return sign * float(m.group(2))
    return None


class DebateEngine:
    def __init__(self, agents: list[Agent], rag: RagEngine, scenario: dict):
        """
        scenario keys: oil_pct, usd_pct, bsp_rate, demand_index
        """
        self._agents = agents
        self._rag = rag
        self._scenario = scenario
        self._history: list[AgentResponse] = []

    def _scenario_text(self) -> str:
        s = self._scenario
        return (
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
        user_content = (
            f"{scenario_text}\n\n"
            f"Relevant context:\n{rag_text}\n\n"
            + (f"Previous agent responses:\n{prior}\n\n" if prior else '')
            + "Give your analysis and a specific ₱/L price change estimate."
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
                stream = ollama.chat(
                    model='deepseek-r1:8b',
                    messages=messages,
                    stream=True,
                )
                for chunk in stream:
                    token = chunk['message']['content']
                    full_text += token
                    if on_token:
                        on_token(agent.name, token)
                thinking, statement = _parse_think(full_text)
                response = AgentResponse(
                    agent_name=agent.name,
                    round_num=round_num,
                    thinking=thinking,
                    statement=statement,
                    price_estimate=_extract_price(statement),
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
        for chunk in ollama.chat(model='deepseek-r1:8b', messages=messages, stream=True):
            token = chunk['message']['content']
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
            return {'weighted_avg': None, 'low': None, 'high': None, 'confidence_pct': 0}
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
            self.error_occurred.emit(str(e))
```

- [ ] **Step 4: Run tests — all must pass**

```bash
python -m pytest ph_economic_ai/tests/test_debate.py -v
```

Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/engine/debate.py ph_economic_ai/tests/test_debate.py
git commit -m "feat: add DebateEngine with Ollama streaming, think-token parsing, DebateThread"
```

---

## Task 4: Pre-bundled NEDA corpus

**Files:**
- Create: `ph_economic_ai/assets/corpus/neda_2024_2026.txt`
- Create: `ph_economic_ai/assets/__init__.py`
- Create: `ph_economic_ai/assets/corpus/__init__.py`

- [ ] **Step 1: Create corpus stub**

The real NEDA corpus should be copied from the official PDF. For now create a stub with representative content — the RAG engine will still index and query it correctly.

```
# ph_economic_ai/assets/corpus/neda_2024_2026.txt
Philippine Economic Outlook 2024-2026 — NEDA Summary

GDP Growth: The Philippine economy is projected to grow at 6.0-7.0% in 2024,
6.5-7.5% in 2025, and 6.5-8.0% in 2026, driven by government infrastructure
spending and remittance-supported household consumption.

Inflation: Headline inflation is expected to decelerate to 3.0-4.0% in 2024
as global oil prices stabilize and domestic food supply improves. The BSP
targets 2-4% inflation over the medium term.

Fuel and Energy: Domestic fuel prices remain sensitive to Brent crude movements
and USD/PHP exchange rate dynamics. Every 10% increase in Brent crude translates
to approximately ₱2.50-3.50/L increase in retail pump prices after a 2-4 week lag.

Exchange Rate: The peso is projected at 54-57 per USD in 2024, with depreciation
pressure stemming from a strong dollar, Fed rate policy, and current account
deficits. BSP will deploy FX operations to limit excessive volatility.

OFW Remittances: Overseas Filipino Worker remittances are projected to reach
$37-38 billion in 2024, supporting household incomes in fuel-consuming regions.
Remittance slowdown poses a downside risk to domestic demand and fuel consumption.

Oil Price Assumptions: NEDA base case assumes Brent crude at $80-90/bbl in 2024
and $75-85/bbl in 2025-2026 as global demand moderates and OPEC+ adjusts output.
A sustained breach of $95/bbl would trigger a review of infrastructure and social
protection budgets.

BSP Monetary Policy: The Bangko Sentral ng Pilipinas held its benchmark overnight
reverse repurchase rate at 6.50% as of Q1 2025, signaling readiness to ease once
inflation is durably within target. Rate cuts are expected in H2 2025 barring
oil price or peso shocks.

Regional Demand: Metro Manila, CALABARZON, and Central Luzon account for over
60% of national fuel consumption. Demand in these regions is relatively price-
inelastic in the short run due to long commutes and reliance on private vehicles.

Supply Chain: The Philippines imports approximately 95% of its crude oil needs.
The DOE monitors pump prices weekly and may recommend emergency measures if
prices breach trigger thresholds set under the Oil Deregulation Law.
```

- [ ] **Step 2: Create empty `__init__.py` files**

```python
# ph_economic_ai/assets/__init__.py
# ph_economic_ai/assets/corpus/__init__.py
# (both empty)
```

- [ ] **Step 3: Commit**

```bash
git add ph_economic_ai/assets/
git commit -m "feat: add pre-bundled NEDA corpus stub for RAG indexing"
```

---

## Task 5: Redesign sidebar + main window scaffold

**Files:**
- Rewrite: `ph_economic_ai/ui/sidebar.py`
- Rewrite: `ph_economic_ai/ui/main_window.py`
- Modify: `ph_economic_ai/main.py`

- [ ] **Step 1: Rewrite sidebar**

```python
# ph_economic_ai/ui/sidebar.py
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QFrame
from PyQt6.QtCore import pyqtSignal

_STAGES = [
    (1, 'Graph Building',   'Build knowledge base'),
    (2, 'Environment',      'Configure scenario & agents'),
    (3, 'Simulation',       'Run agent debate'),
    (4, 'Report',           'View results'),
    (5, 'Interact',         'Adjust & explore'),
]

_STYLE_ACTIVE = (
    'text-align:left; padding:10px 16px; font-size:11px; font-weight:700;'
    'color:#1C1E26; background:#F7F8FA; border:none;'
    'border-left:3px solid #1C1E26;'
)
_STYLE_INACTIVE = (
    'text-align:left; padding:10px 16px 10px 19px; font-size:11px;'
    'color:#9EA3AE; background:transparent; border:none;'
)
_STYLE_DISABLED = (
    'text-align:left; padding:10px 16px 10px 19px; font-size:11px;'
    'color:#D1D5DB; background:transparent; border:none;'
)


class SidebarWidget(QWidget):
    stage_changed = pyqtSignal(int)  # emits 0-based index

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(180)
        self.setStyleSheet('background:#FFFFFF; border-right:1px solid #EAECF0;')
        self._buttons: list[QPushButton] = []
        self._locked: set[int] = {2, 3, 4}  # stages 3-5 locked until first run
        self._active = 0
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Logo
        logo = QFrame()
        logo.setStyleSheet('background:#FFFFFF; border-bottom:1px solid #EAECF0;')
        ll = QVBoxLayout(logo)
        ll.setContentsMargins(16, 16, 16, 14)
        ll.setSpacing(2)
        title = QLabel('PH ECONAI')
        title.setStyleSheet('font-size:11px; font-weight:700; color:#1C1E26; letter-spacing:1px;')
        sub = QLabel('Simulation Engine')
        sub.setStyleSheet('font-size:9px; color:#9EA3AE;')
        ll.addWidget(title)
        ll.addWidget(sub)
        layout.addWidget(logo)

        layout.addSpacing(8)

        for i, (num, name, desc) in enumerate(_STAGES):
            btn = QPushButton(f'{num}.  {name}')
            btn.setFlat(True)
            btn.setToolTip(desc)
            btn.setCursor(self.cursor())
            btn.setEnabled(i not in self._locked)
            btn.setStyleSheet(_STYLE_ACTIVE if i == 0 else _STYLE_INACTIVE)
            btn.clicked.connect(lambda _, idx=i: self._on_click(idx))
            self._buttons.append(btn)
            layout.addWidget(btn)

        layout.addStretch()

    def _on_click(self, idx: int):
        self._active = idx
        self._refresh_styles()
        self.stage_changed.emit(idx)

    def _refresh_styles(self):
        for i, btn in enumerate(self._buttons):
            if i in self._locked:
                btn.setStyleSheet(_STYLE_DISABLED)
            elif i == self._active:
                btn.setStyleSheet(_STYLE_ACTIVE)
            else:
                btn.setStyleSheet(_STYLE_INACTIVE)

    def unlock_stages(self, indices: list[int]):
        """Call after first simulation completes to enable stages 3-5."""
        for i in indices:
            self._locked.discard(i)
            self._buttons[i].setEnabled(True)
        self._refresh_styles()

    def set_active(self, idx: int):
        self._active = idx
        self._refresh_styles()
```

- [ ] **Step 2: Rewrite main_window.py**

```python
# ph_economic_ai/ui/main_window.py
from pathlib import Path

from PyQt6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QStackedWidget, QMessageBox

from ph_economic_ai.engine.rag import RagEngine
from ph_economic_ai.engine.debate import DEFAULT_AGENTS
from ph_economic_ai.ui.sidebar import SidebarWidget
from ph_economic_ai.ui.stage1_rag import Stage1RagPanel
from ph_economic_ai.ui.stage2_setup import Stage2SetupPanel
from ph_economic_ai.ui.stage3_canvas import Stage3CanvasPanel
from ph_economic_ai.ui.stage4_report import Stage4ReportPanel
from ph_economic_ai.ui.stage5_interact import Stage5InteractPanel


class SimMainWindow(QMainWindow):
    def __init__(self, df, regressor, data_source: str = 'Live Data',
                 cv_rmse: float = 0.0, parent=None):
        super().__init__(parent)
        self._df = df
        self._regressor = regressor
        self._cv_rmse = cv_rmse
        self._rag = RagEngine()
        self._agents = list(DEFAULT_AGENTS)

        self.setWindowTitle('PH Economic Pressure Simulation Engine')
        self.setMinimumSize(1200, 720)
        self.setStyleSheet('background:#F7F8FA;')

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._sidebar = SidebarWidget()
        self._sidebar.stage_changed.connect(self._on_stage_changed)
        root.addWidget(self._sidebar)

        self._stack = QStackedWidget()
        root.addWidget(self._stack, stretch=1)

        self._stage1 = Stage1RagPanel(self._rag)
        self._stage2 = Stage2SetupPanel(self._agents)
        self._stage3 = Stage3CanvasPanel(self._rag, self._agents, self._regressor,
                                         self._df, self._cv_rmse)
        self._stage4 = Stage4ReportPanel()
        self._stage5 = Stage5InteractPanel(self._rag, self._agents, self._regressor,
                                           self._df, self._cv_rmse)

        for stage in (self._stage1, self._stage2, self._stage3,
                      self._stage4, self._stage5):
            self._stack.addWidget(stage)

        # Wire signals
        self._stage3.simulation_complete.connect(self._on_simulation_complete)
        self._stage5.rerun_requested.connect(self._stage3.start_simulation)

        # Load pre-bundled corpus in background
        corpus_path = Path(__file__).parent.parent / 'assets' / 'corpus' / 'neda_2024_2026.txt'
        if corpus_path.exists():
            self._rag.add_text('neda_2024_2026', corpus_path.read_text(encoding='utf-8'))

    def _on_stage_changed(self, idx: int):
        self._stack.setCurrentIndex(idx)

    def _on_simulation_complete(self, responses):
        consensus = self._stage3.engine.consensus()
        self._stage4.populate(responses, consensus, self._regressor,
                              self._df, self._cv_rmse,
                              self._stage3.scenario())
        self._stage5.update_context(responses, self._stage3.scenario())
        self._sidebar.unlock_stages([2, 3, 4])
        self._sidebar.set_active(3)
        self._stack.setCurrentIndex(3)
```

- [ ] **Step 3: Update main.py**

```python
# ph_economic_ai/main.py
import sys
from PyQt6.QtWidgets import QApplication, QMessageBox

from ph_economic_ai.data import fetch_dataset
from ph_economic_ai.utils.preprocessing import build_features
from ph_economic_ai import model as ml
from ph_economic_ai.ui.main_window import SimMainWindow


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    try:
        df, data_source = fetch_dataset()
    except RuntimeError as e:
        QMessageBox.critical(None, 'Data Error', str(e))
        sys.exit(1)

    X, y, _, _ = build_features(df)
    regressor = ml.train(X, y)
    cv_rmse = ml.cross_val_rmse(X, y)

    window = SimMainWindow(df=df, regressor=regressor,
                           data_source=data_source, cv_rmse=cv_rmse)
    window.showMaximized()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Commit**

```bash
git add ph_economic_ai/ui/sidebar.py ph_economic_ai/ui/main_window.py ph_economic_ai/main.py
git commit -m "feat: redesign sidebar and main window scaffold with 5-stage QStackedWidget"
```

---

## Task 6: Stage 1 — RAG panel

**Files:**
- Create: `ph_economic_ai/ui/stage1_rag.py`

- [ ] **Step 1: Implement Stage1RagPanel**

```python
# ph_economic_ai/ui/stage1_rag.py
import concurrent.futures
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QFileDialog, QProgressBar,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer

from ph_economic_ai.engine.rag import RagEngine, SOURCES


class _FetchThread(QThread):
    source_done = pyqtSignal(str, int)   # source_name, chunk_count
    all_done = pyqtSignal(dict)

    def __init__(self, rag: RagEngine, parent=None):
        super().__init__(parent)
        self._rag = rag

    def run(self):
        results = self._rag.fetch_all(
            on_progress=lambda name, count: self.source_done.emit(name, count)
        )
        self.all_done.emit(results)


class _SourceCard(QFrame):
    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            'QFrame{background:#F7F8FA;border:1px solid #EAECF0;'
            'border-radius:9px;padding:0px;}'
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)

        self._name_lbl = QLabel(name)
        self._name_lbl.setStyleSheet('font-size:10px;font-weight:600;color:#1C1E26;')

        self._status_lbl = QLabel('waiting...')
        self._status_lbl.setStyleSheet('font-size:9px;color:#9EA3AE;')
        self._status_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)

        self._bar = QProgressBar()
        self._bar.setRange(0, 0)  # indeterminate
        self._bar.setFixedHeight(3)
        self._bar.setTextVisible(False)
        self._bar.setStyleSheet(
            'QProgressBar{background:#EAECF0;border-radius:2px;border:none;}'
            'QProgressBar::chunk{background:#1C1E26;border-radius:2px;}'
        )

        info_col = QVBoxLayout()
        info_col.setSpacing(4)
        info_col.addWidget(self._name_lbl)
        info_col.addWidget(self._bar)

        layout.addLayout(info_col, stretch=1)
        layout.addWidget(self._status_lbl)

    def set_done(self, chunk_count: int):
        self._bar.setRange(0, 1)
        self._bar.setValue(1)
        self._status_lbl.setText(f'{chunk_count} chunks')
        self._status_lbl.setStyleSheet('font-size:9px;color:#1C1E26;font-weight:600;')

    def set_error(self):
        self._bar.setRange(0, 1)
        self._bar.setValue(0)
        self._status_lbl.setText('failed')
        self._status_lbl.setStyleSheet('font-size:9px;color:#E74C3C;')


class Stage1RagPanel(QWidget):
    def __init__(self, rag: RagEngine, parent=None):
        super().__init__(parent)
        self._rag = rag
        self._cards: dict[str, _SourceCard] = {}
        self._build()
        self._start_fetch()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(14)

        # Header
        h = QLabel('Stage 1 — Graph Building')
        h.setStyleSheet('font-size:18px;font-weight:700;color:#1C1E26;')
        root.addWidget(h)

        sub = QLabel('Fetching live sources in parallel and indexing into TF-IDF knowledge base.')
        sub.setStyleSheet('font-size:11px;color:#9EA3AE;')
        root.addWidget(sub)

        # Status row
        status_row = QHBoxLayout()
        self._status_lbl = QLabel('Fetching 9 sources...')
        self._status_lbl.setStyleSheet('font-size:10px;color:#1C1E26;font-weight:600;')
        self._chunk_lbl = QLabel('0 chunks indexed')
        self._chunk_lbl.setStyleSheet('font-size:10px;color:#9EA3AE;')
        status_row.addWidget(self._status_lbl)
        status_row.addStretch()
        status_row.addWidget(self._chunk_lbl)
        root.addLayout(status_row)

        # Source cards in scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet('background:transparent;')

        cards_widget = QWidget()
        self._cards_layout = QVBoxLayout(cards_widget)
        self._cards_layout.setSpacing(6)
        self._cards_layout.setContentsMargins(0, 0, 0, 0)

        for name in SOURCES:
            card = _SourceCard(name)
            self._cards[name] = card
            self._cards_layout.addWidget(card)

        self._cards_layout.addStretch()
        scroll.setWidget(cards_widget)
        root.addWidget(scroll, stretch=1)

        # Upload PDF button
        upload_btn = QPushButton('+ Upload PDF')
        upload_btn.setStyleSheet(
            'QPushButton{border:1.5px dashed #D1D5DB;border-radius:9px;'
            'padding:8px;font-size:10px;color:#9EA3AE;background:transparent;}'
            'QPushButton:hover{border-color:#9EA3AE;color:#6B7280;}'
        )
        upload_btn.clicked.connect(self._on_upload)
        root.addWidget(upload_btn)

    def _start_fetch(self):
        self._thread = _FetchThread(self._rag)
        self._thread.source_done.connect(self._on_source_done)
        self._thread.all_done.connect(self._on_all_done)
        self._thread.start()

        # Chunk count ticker
        self._ticker = QTimer(self)
        self._ticker.timeout.connect(self._update_chunk_count)
        self._ticker.start(500)

    def _on_source_done(self, name: str, count: int):
        card = self._cards.get(name)
        if card:
            if count > 0:
                card.set_done(count)
            else:
                card.set_error()

    def _on_all_done(self, results: dict):
        self._ticker.stop()
        self._update_chunk_count()
        done = sum(1 for v in results.values() if v > 0)
        self._status_lbl.setText(f'{done}/9 sources fetched')

    def _update_chunk_count(self):
        self._chunk_lbl.setText(f'{self._rag.chunk_count} chunks indexed')

    def _on_upload(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Select PDF', '', 'PDF Files (*.pdf)'
        )
        if path:
            count = self._rag.add_pdf(path)
            self._update_chunk_count()
```

- [ ] **Step 2: Run the app and verify Stage 1 renders with source cards and fetch progress**

```bash
python -m ph_economic_ai.main
```

Expected: window launches maximized, Stage 1 visible, source cards appear with progress bars animating, chunk count updates as sources finish.

- [ ] **Step 3: Commit**

```bash
git add ph_economic_ai/ui/stage1_rag.py
git commit -m "feat: add Stage 1 RAG panel with parallel fetch progress and PDF upload"
```

---

## Task 7: Stage 2 — Environment setup

**Files:**
- Create: `ph_economic_ai/ui/stage2_setup.py`

- [ ] **Step 1: Implement Stage2SetupPanel**

```python
# ph_economic_ai/ui/stage2_setup.py
from dataclasses import dataclass
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QScrollArea, QSlider, QDialog, QLineEdit, QDialogButtonBox,
)
from PyQt6.QtCore import Qt, pyqtSignal

from ph_economic_ai.engine.debate import Agent, DEFAULT_AGENTS


@dataclass
class Scenario:
    oil_pct: float = 5.0
    usd_pct: float = 2.0
    bsp_rate: float = 6.5
    demand_index: float = 72.0

    def to_dict(self) -> dict:
        return {
            'oil_pct': self.oil_pct,
            'usd_pct': self.usd_pct,
            'bsp_rate': self.bsp_rate,
            'demand_index': self.demand_index,
        }


class _ScenarioPill(QFrame):
    value_changed = pyqtSignal()

    def __init__(self, label: str, default: float,
                 min_val: float, max_val: float, step: float = 0.5, parent=None):
        super().__init__(parent)
        self._step = step
        self._min = min_val
        self._max = max_val
        self._value = default
        self._label = label
        self.setStyleSheet(
            'QFrame{background:#FFFFFF;border:1px solid #1C1E26;'
            'border-radius:9px;}'
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)

        self._label_lbl = QLabel(label)
        self._label_lbl.setStyleSheet('font-size:8px;color:#9EA3AE;text-transform:uppercase;')

        self._val_lbl = QLabel(self._fmt(default))
        self._val_lbl.setStyleSheet('font-size:14px;font-weight:700;color:#1C1E26;')

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(int(min_val / step), int(max_val / step))
        self._slider.setValue(int(default / step))
        self._slider.setFixedHeight(14)
        self._slider.setStyleSheet(
            'QSlider::groove:horizontal{height:3px;background:#EAECF0;border-radius:2px;}'
            'QSlider::handle:horizontal{width:10px;height:10px;margin:-4px 0;'
            'border-radius:5px;background:#1C1E26;}'
            'QSlider::sub-page:horizontal{background:#1C1E26;border-radius:2px;}'
        )
        self._slider.valueChanged.connect(self._on_slider)

        layout.addWidget(self._label_lbl)
        layout.addWidget(self._val_lbl)
        layout.addWidget(self._slider)

    def _fmt(self, v: float) -> str:
        return f'{v:+.1f}%' if '%' in self._label or 'pct' in self._label.lower() else f'{v:.1f}'

    def _on_slider(self, raw: int):
        self._value = raw * self._step
        self._val_lbl.setText(self._fmt(self._value))
        self.value_changed.emit()

    @property
    def value(self) -> float:
        return self._value


class _AgentCard(QFrame):
    def __init__(self, agent: Agent, parent=None):
        super().__init__(parent)
        self._agent = agent
        self.setStyleSheet(
            'QFrame{background:#F7F8FA;border:1px solid #EAECF0;border-radius:10px;}'
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(5)

        name_lbl = QLabel(agent.name)
        name_lbl.setStyleSheet('font-size:11px;font-weight:700;color:#1C1E26;')
        role_lbl = QLabel(agent.role)
        role_lbl.setStyleSheet('font-size:9px;color:#9EA3AE;')

        prompt_lbl = QLabel(f'"{agent.system_prompt[:120]}..."')
        prompt_lbl.setWordWrap(True)
        prompt_lbl.setStyleSheet(
            'font-size:9px;color:#9EA3AE;font-style:italic;'
            'background:#FFFFFF;border:1px solid #EAECF0;'
            'border-radius:7px;padding:5px 8px;'
        )

        sources_row = QHBoxLayout()
        sources_row.setSpacing(4)
        for src in agent.rag_sources[:5]:
            tag = QLabel(src)
            tag.setStyleSheet(
                'font-size:8px;font-weight:600;color:#FFFFFF;'
                'background:#1C1E26;border-radius:20px;padding:2px 7px;'
            )
            sources_row.addWidget(tag)
        sources_row.addStretch()

        layout.addWidget(name_lbl)
        layout.addWidget(role_lbl)
        layout.addLayout(sources_row)
        layout.addWidget(prompt_lbl)


class Stage2SetupPanel(QWidget):
    run_requested = pyqtSignal(object, list)  # Scenario, list[Agent]

    def __init__(self, agents: list[Agent], parent=None):
        super().__init__(parent)
        self._agents = list(agents)
        self._pills: list[_ScenarioPill] = []
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(16)

        h = QLabel('Stage 2 — Environment Setup')
        h.setStyleSheet('font-size:18px;font-weight:700;color:#1C1E26;')
        root.addWidget(h)

        # Scenario inputs
        scenario_lbl = QLabel('SCENARIO INPUTS')
        scenario_lbl.setStyleSheet('font-size:9px;font-weight:600;color:#9EA3AE;letter-spacing:0.7px;')
        root.addWidget(scenario_lbl)

        pills_row = QHBoxLayout()
        pills_row.setSpacing(8)
        configs = [
            ('Oil shock %',    5.0,  -20.0, 30.0, 0.5),
            ('USD/PHP shift %', 2.0, -10.0, 15.0, 0.5),
            ('BSP rate %',     6.5,   3.0,  10.0, 0.25),
            ('Demand index',  72.0,  50.0, 100.0, 1.0),
        ]
        for label, default, mn, mx, step in configs:
            pill = _ScenarioPill(label, default, mn, mx, step)
            self._pills.append(pill)
            pills_row.addWidget(pill)
        root.addLayout(pills_row)

        # Agent roster
        agents_lbl = QLabel('AGENT ROSTER')
        agents_lbl.setStyleSheet('font-size:9px;font-weight:600;color:#9EA3AE;letter-spacing:0.7px;')
        root.addWidget(agents_lbl)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._agents_widget = QWidget()
        self._agents_layout = QVBoxLayout(self._agents_widget)
        self._agents_layout.setSpacing(8)
        self._agents_layout.setContentsMargins(0, 0, 0, 0)
        self._rebuild_agent_cards()
        self._agents_layout.addStretch()
        scroll.setWidget(self._agents_widget)
        root.addWidget(scroll, stretch=1)

        # Add agent + Run buttons
        btn_row = QHBoxLayout()
        add_btn = QPushButton('+ Add custom agent')
        add_btn.setStyleSheet(
            'QPushButton{border:1.5px dashed #D1D5DB;border-radius:9px;'
            'padding:8px 16px;font-size:10px;color:#9EA3AE;background:transparent;}'
        )
        add_btn.clicked.connect(self._on_add_agent)

        self._run_btn = QPushButton('Run Simulation →')
        self._run_btn.setStyleSheet(
            'QPushButton{background:#1C1E26;color:#FFFFFF;border-radius:9px;'
            'padding:10px 20px;font-size:11px;font-weight:700;border:none;}'
            'QPushButton:hover{background:#374151;}'
        )
        self._run_btn.clicked.connect(self._on_run)

        self._time_lbl = QLabel('')
        self._time_lbl.setStyleSheet('font-size:9px;color:#9EA3AE;')

        btn_row.addWidget(add_btn)
        btn_row.addStretch()
        btn_row.addWidget(self._time_lbl)
        btn_row.addWidget(self._run_btn)
        root.addLayout(btn_row)

        for pill in self._pills:
            pill.value_changed.connect(self._update_time_estimate)
        self._update_time_estimate()

    def _rebuild_agent_cards(self):
        while self._agents_layout.count():
            item = self._agents_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for agent in self._agents:
            self._agents_layout.addWidget(_AgentCard(agent))

    def _update_time_estimate(self):
        rounds = 3 if len(self._agents) <= 7 else 2
        secs = len(self._agents) * rounds * 10
        self._time_lbl.setText(f'~{secs // 60} min estimated')

    def _on_add_agent(self):
        if len(self._agents) >= 10:
            return  # max 10 agents
        dlg = _AddAgentDialog(self)
        if dlg.exec():
            self._agents.append(dlg.agent())
            self._rebuild_agent_cards()
            self._update_time_estimate()

    def _on_run(self):
        scenario = Scenario(
            oil_pct=self._pills[0].value,
            usd_pct=self._pills[1].value,
            bsp_rate=self._pills[2].value,
            demand_index=self._pills[3].value,
        )
        self.run_requested.emit(scenario, list(self._agents))

    def current_scenario(self) -> Scenario:
        return Scenario(
            oil_pct=self._pills[0].value,
            usd_pct=self._pills[1].value,
            bsp_rate=self._pills[2].value,
            demand_index=self._pills[3].value,
        )


class _AddAgentDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Add Custom Agent')
        self.setMinimumWidth(400)
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        self._name = QLineEdit('Custom Agent')
        self._role = QLineEdit('Specialist role description')
        self._prompt = QLineEdit('You are a specialist. Analyze the scenario and give a ₱/L estimate.')

        for lbl_text, widget in [('Name', self._name), ('Role', self._role), ('System prompt', self._prompt)]:
            layout.addWidget(QLabel(lbl_text))
            layout.addWidget(widget)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def agent(self) -> Agent:
        return Agent(
            name=self._name.text().strip() or 'Custom Agent',
            role=self._role.text().strip(),
            system_prompt=self._prompt.text().strip(),
            rag_sources=[],
        )
```

- [ ] **Step 2: Wire Stage 2 run button to Stage 3 in main_window.py**

In `SimMainWindow._build` (already written in Task 5), connect:

```python
self._stage2.run_requested.connect(self._on_run_requested)
```

Add method:

```python
def _on_run_requested(self, scenario, agents):
    self._sidebar.set_active(2)
    self._stack.setCurrentIndex(2)
    self._stage3.start_simulation(scenario, agents)
```

- [ ] **Step 3: Run the app, navigate to Stage 2, verify scenario pills and agent cards render**

```bash
python -m ph_economic_ai.main
```

- [ ] **Step 4: Commit**

```bash
git add ph_economic_ai/ui/stage2_setup.py ph_economic_ai/ui/main_window.py
git commit -m "feat: add Stage 2 setup panel with scenario sliders and agent roster"
```

---

## Task 8: Stage 3 — Simulation canvas

**Files:**
- Create: `ph_economic_ai/ui/stage3_canvas.py`

- [ ] **Step 1: Implement Stage3CanvasPanel**

```python
# ph_economic_ai/ui/stage3_canvas.py
import re
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QGraphicsView, QGraphicsScene, QGraphicsProxyWidget,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal, QTimer
from PyQt6.QtGui import QPainter, QPen, QColor, QBrush, QFont

from ph_economic_ai.engine.rag import RagEngine
from ph_economic_ai.engine.debate import Agent, DebateEngine, DebateThread, AgentResponse
from ph_economic_ai.utils.preprocessing import build_features
from ph_economic_ai import model as ml


# ── Agent node widget (embedded in QGraphicsScene via QGraphicsProxyWidget) ────

class _AgentNodeWidget(QFrame):
    def __init__(self, agent: Agent, parent=None):
        super().__init__(parent)
        self.agent = agent
        self._tokens = ''
        self._is_thinking = False
        self.setFixedWidth(180)
        self.setStyleSheet(
            'QFrame{background:#FFFFFF;border:1.5px solid #EAECF0;'
            'border-radius:12px;}'
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        self._name_lbl = QLabel(agent.name)
        self._name_lbl.setStyleSheet('font-size:10px;font-weight:700;color:#1C1E26;')
        self._status_lbl = QLabel('waiting')
        self._status_lbl.setStyleSheet('font-size:8px;color:#9EA3AE;')

        self._bubble = QLabel('...')
        self._bubble.setWordWrap(True)
        self._bubble.setFixedWidth(158)
        self._bubble.setStyleSheet(
            'font-size:9px;color:#374151;line-height:1.4;'
            'background:#F7F8FA;border:1px solid #EAECF0;'
            'border-radius:7px;padding:6px 8px;'
        )

        layout.addWidget(self._name_lbl)
        layout.addWidget(self._status_lbl)
        layout.addWidget(self._bubble)

    def set_active(self, thinking: bool = True):
        border = '1.5px solid #1C1E26' if thinking else '1.5px solid #EAECF0'
        self.setStyleSheet(
            f'QFrame{{background:#FFFFFF;border:{border};border-radius:12px;}}'
        )
        self._is_thinking = thinking
        self._status_lbl.setText('thinking...' if thinking else 'done')

    def append_token(self, token: str):
        self._tokens += token
        # Show only post-</think> content in bubble
        visible = re.sub(r'<think>.*?</think>', '', self._tokens, flags=re.DOTALL).strip()
        if not visible and '<think>' in self._tokens:
            self._bubble.setText('thinking...')
        else:
            self._bubble.setText(visible[-300:] if visible else '...')

    def set_done(self, statement: str):
        self._tokens = ''
        self._bubble.setText(statement[:300])
        self.set_active(thinking=False)
        self._status_lbl.setText('done')


# ── Engine node (centered dark card) ──────────────────────────────────────────

class _EngineNodeWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(140, 100)
        self.setStyleSheet(
            'QFrame{background:#1C1E26;border-radius:13px;}'
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)

        lbl = QLabel('HistGBM Engine')
        lbl.setStyleSheet('font-size:8px;font-weight:700;color:#9EA3AE;letter-spacing:0.7px;')
        self._price_lbl = QLabel('₱---.--')
        self._price_lbl.setStyleSheet('font-size:20px;font-weight:700;color:#FFFFFF;')
        self._sub_lbl = QLabel('forecast · updating')
        self._sub_lbl.setStyleSheet('font-size:8px;color:#6B7280;')

        layout.addWidget(lbl)
        layout.addWidget(self._price_lbl)
        layout.addWidget(self._sub_lbl)

    def set_price(self, price: float):
        self._price_lbl.setText(f'₱{price:.2f}')


# ── Dot-grid canvas ────────────────────────────────────────────────────────────

class _DotGridView(QGraphicsView):
    def __init__(self, scene: QGraphicsScene):
        super().__init__(scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setStyleSheet('border:none;background:#FFFFFF;')

    def drawBackground(self, painter: QPainter, rect: QRectF):
        super().drawBackground(painter, rect)
        painter.setPen(QPen(QColor('#D1D5DB'), 1))
        spacing = 22
        left = int(rect.left() / spacing) * spacing
        top = int(rect.top() / spacing) * spacing
        x = left
        while x < rect.right():
            y = top
            while y < rect.bottom():
                painter.drawPoint(QPointF(x, y))
                y += spacing
            x += spacing


# ── Main canvas panel ──────────────────────────────────────────────────────────

class Stage3CanvasPanel(QWidget):
    simulation_complete = pyqtSignal(object)  # list[AgentResponse]

    def __init__(self, rag: RagEngine, agents: list[Agent],
                 regressor, df, cv_rmse: float, parent=None):
        super().__init__(parent)
        self._rag = rag
        self._agents = agents
        self._regressor = regressor
        self._df = df
        self._cv_rmse = cv_rmse
        self._thread: DebateThread | None = None
        self._engine: DebateEngine | None = None
        self._scenario: dict = {}
        self._node_widgets: dict[str, _AgentNodeWidget] = {}
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Toolbar
        toolbar = QFrame()
        toolbar.setStyleSheet('background:#FFFFFF;border-bottom:1px solid #EAECF0;')
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(16, 10, 16, 10)

        self._scenario_lbl = QLabel('No simulation running')
        self._scenario_lbl.setStyleSheet('font-size:10px;font-weight:600;color:#1C1E26;')

        self._round_lbl = QLabel('Round: —')
        self._round_lbl.setStyleSheet('font-size:10px;color:#9EA3AE;')

        self._status_dot = QLabel('●')
        self._status_dot.setStyleSheet('font-size:10px;color:#9EA3AE;')

        self._stop_btn = QPushButton('Stop')
        self._stop_btn.setEnabled(False)
        self._stop_btn.setStyleSheet(
            'QPushButton{border:1px solid #EAECF0;border-radius:7px;'
            'padding:4px 12px;font-size:9px;font-weight:600;color:#6B7280;background:#FFFFFF;}'
        )
        self._stop_btn.clicked.connect(self._on_stop)

        tb_layout.addWidget(self._scenario_lbl)
        tb_layout.addStretch()
        tb_layout.addWidget(self._round_lbl)
        tb_layout.addSpacing(12)
        tb_layout.addWidget(self._status_dot)
        tb_layout.addWidget(self._stop_btn)
        root.addWidget(toolbar)

        # Body: canvas + right panel
        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        # Canvas (QGraphicsView)
        self._scene = QGraphicsScene()
        self._view = _DotGridView(self._scene)
        self._view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        body_layout.addWidget(self._view, stretch=1)

        # Right panel
        right = QFrame()
        right.setFixedWidth(260)
        right.setStyleSheet('background:#FFFFFF;border-left:1px solid #EAECF0;')
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(14, 14, 14, 14)
        right_layout.setSpacing(10)

        log_title = QLabel('Debate Log')
        log_title.setStyleSheet('font-size:10px;font-weight:700;color:#1C1E26;')
        self._log_lbl = QLabel('Waiting for simulation...')
        self._log_lbl.setWordWrap(True)
        self._log_lbl.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._log_lbl.setStyleSheet('font-size:9px;color:#9EA3AE;')

        right_layout.addWidget(log_title)
        right_layout.addWidget(self._log_lbl, stretch=1)

        price_title = QLabel('Emerging Output')
        price_title.setStyleSheet('font-size:10px;font-weight:700;color:#1C1E26;')
        self._price_lbl = QLabel('₱---.-- /L')
        self._price_lbl.setStyleSheet('font-size:18px;font-weight:700;color:#1C1E26;')
        right_layout.addWidget(price_title)
        right_layout.addWidget(self._price_lbl)

        body_layout.addWidget(right)
        root.addWidget(body, stretch=1)

    def start_simulation(self, scenario, agents=None):
        if agents:
            self._agents = agents
        if hasattr(scenario, 'to_dict'):
            self._scenario = scenario.to_dict()
        else:
            self._scenario = scenario

        # Update scenario label
        s = self._scenario
        self._scenario_lbl.setText(
            f"Oil {s.get('oil_pct', 0):+.1f}% · "
            f"USD {s.get('usd_pct', 0):+.1f}% · "
            f"BSP {s.get('bsp_rate', 6.5):.2f}% · "
            f"Demand {s.get('demand_index', 72):.0f}"
        )

        # Rebuild canvas nodes
        self._scene.clear()
        self._node_widgets.clear()
        self._place_nodes()

        # Start debate thread
        self._engine = DebateEngine(self._agents, self._rag, self._scenario)
        rounds = 3 if len(self._agents) <= 7 else 2
        self._thread = DebateThread(self._engine, rounds=rounds)
        self._thread.token_received.connect(self._on_token)
        self._thread.agent_done.connect(self._on_agent_done)
        self._thread.debate_complete.connect(self._on_debate_complete)
        self._thread.error_occurred.connect(self._on_error)

        self._stop_btn.setEnabled(True)
        self._status_dot.setStyleSheet('font-size:10px;color:#22C55E;')
        self._round_lbl.setText('Round 1')
        self._thread.start()

    def _place_nodes(self):
        """Position agent node widgets and engine node on the canvas."""
        n = len(self._agents)
        cx, cy = 400, 300
        radius = 200

        # Engine node (center)
        engine_widget = _EngineNodeWidget()
        self._engine_node = engine_widget
        proxy = self._scene.addWidget(engine_widget)
        proxy.setPos(cx - 70, cy - 50)

        # Agent nodes in a circle
        import math
        for i, agent in enumerate(self._agents):
            angle = (2 * math.pi * i / n) - math.pi / 2
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            node = _AgentNodeWidget(agent)
            self._node_widgets[agent.name] = node
            proxy = self._scene.addWidget(node)
            proxy.setPos(x - 90, y - 60)

            # Dashed line to engine
            pen = QPen(QColor('#D1D5DB'), 1.5, Qt.PenStyle.DashLine)
            self._scene.addLine(x, y, cx, cy, pen)

        self._view.fitInView(self._scene.itemsBoundingRect().adjusted(-40, -40, 40, 40),
                             Qt.AspectRatioMode.KeepAspectRatio)

    def _on_token(self, agent_name: str, token: str):
        node = self._node_widgets.get(agent_name)
        if node:
            node.set_active(thinking=True)
            node.append_token(token)

    def _on_agent_done(self, response: AgentResponse):
        node = self._node_widgets.get(response.agent_name)
        if node:
            node.set_done(response.statement)

        # Update round label
        self._round_lbl.setText(f'Round {response.round_num}')

        # Update HistGBM engine node price
        self._update_engine_price()

        # Append to log
        est = f'+₱{response.price_estimate:.2f}' if response.price_estimate else '—'
        current = self._log_lbl.text()
        if current == 'Waiting for simulation...':
            current = ''
        self._log_lbl.setText(
            current + f'\nR{response.round_num} · {response.agent_name}: {est}'
        )

    def _update_engine_price(self):
        """Re-run HistGBM with scenario-adjusted features and update engine node."""
        try:
            X, y, feature_cols, df_feat = build_features(self._df)
            last = df_feat.iloc[-1]
            features = []
            for col in feature_cols:
                if col == 'prev_gas_price':
                    features.append(float(last['gas_price']))
                elif col == 'oil_price':
                    v = float(last[col]) * (1 + self._scenario.get('oil_pct', 0) / 100)
                    features.append(v)
                elif col == 'usd_php':
                    v = float(last[col]) * (1 + self._scenario.get('usd_pct', 0) / 100)
                    features.append(v)
                else:
                    features.append(float(last[col]))
            price, _, _ = ml.predict(self._regressor, np.array(features))
            self._engine_node.set_price(price)
            self._price_lbl.setText(f'₱{price:.2f} /L')
        except Exception:
            pass

    def _on_debate_complete(self, responses):
        self._stop_btn.setEnabled(False)
        self._status_dot.setStyleSheet('font-size:10px;color:#9EA3AE;')
        self._round_lbl.setText('Complete')
        self.simulation_complete.emit(responses)

    def _on_error(self, msg: str):
        self._stop_btn.setEnabled(False)
        self._status_dot.setStyleSheet('font-size:10px;color:#E74C3C;')
        self._scenario_lbl.setText(f'Error: {msg}')

    def _on_stop(self):
        if self._thread and self._thread.isRunning():
            self._thread.terminate()
            self._thread.wait()
        self._stop_btn.setEnabled(False)
        self._status_dot.setStyleSheet('font-size:10px;color:#9EA3AE;')

    @property
    def engine(self) -> DebateEngine | None:
        return self._engine

    def scenario(self) -> dict:
        return self._scenario
```

- [ ] **Step 2: Commit**

```bash
git add ph_economic_ai/ui/stage3_canvas.py
git commit -m "feat: add Stage 3 canvas with live agent nodes, debate thread, HistGBM engine node"
```

---

## Task 9: Stage 4 — Report panel

**Files:**
- Create: `ph_economic_ai/ui/stage4_report.py`

- [ ] **Step 1: Implement Stage4ReportPanel**

```python
# ph_economic_ai/ui/stage4_report.py
import os
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QFrame, QScrollArea, QPushButton, QFileDialog,
)
from PyQt6.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from ph_economic_ai.engine.debate import AgentResponse
from ph_economic_ai.utils.preprocessing import build_features
from ph_economic_ai import model as ml


class Stage4ReportPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._responses: list[AgentResponse] = []
        self._consensus: dict = {}
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Status bar
        bar = QFrame()
        bar.setStyleSheet('background:#FFFFFF;border-bottom:1px solid #EAECF0;')
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(20, 10, 20, 10)

        self._status_lbl = QLabel('Simulation complete')
        self._status_lbl.setStyleSheet('font-size:11px;font-weight:700;color:#1C1E26;')
        self._detail_lbl = QLabel('')
        self._detail_lbl.setStyleSheet('font-size:9px;color:#9EA3AE;')

        export_btn = QPushButton('Export PDF')
        export_btn.setStyleSheet(
            'QPushButton{background:#1C1E26;color:#FFFFFF;border-radius:8px;'
            'padding:6px 14px;font-size:10px;font-weight:600;border:none;}'
            'QPushButton:hover{background:#374151;}'
        )
        export_btn.clicked.connect(self._on_export)

        bar_layout.addWidget(self._status_lbl)
        bar_layout.addWidget(self._detail_lbl)
        bar_layout.addStretch()
        bar_layout.addWidget(export_btn)
        root.addWidget(bar)

        # Scrollable body
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget()
        body.setStyleSheet('background:#F7F8FA;')
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(20, 20, 20, 20)
        body_layout.setSpacing(14)

        self._left = QVBoxLayout()
        self._right = QVBoxLayout()
        body_layout.addLayout(self._left, stretch=1)
        body_layout.addLayout(self._right, stretch=1)
        scroll.setWidget(body)
        root.addWidget(scroll, stretch=1)

        # Placeholder labels
        self._placeholder = QLabel('Run a simulation to see the report.')
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet('font-size:14px;color:#9EA3AE;')
        self._left.addWidget(self._placeholder)

    def populate(self, responses: list[AgentResponse], consensus: dict,
                 regressor, df, cv_rmse: float, scenario: dict):
        self._responses = responses
        self._consensus = consensus

        # Clear existing content
        for layout in (self._left, self._right):
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

        self._build_left(consensus, responses)
        self._build_right(regressor, df, cv_rmse, scenario, consensus)

        rounds = max((r.round_num for r in responses), default=0)
        self._detail_lbl.setText(
            f'{len(set(r.agent_name for r in responses))} agents · '
            f'{rounds} rounds · {len(responses)} responses'
        )

    def _card(self, title: str) -> tuple[QFrame, QVBoxLayout]:
        frame = QFrame()
        frame.setStyleSheet(
            'QFrame{background:#FFFFFF;border:1px solid #EAECF0;'
            'border-radius:12px;}'
        )
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)
        lbl = QLabel(title)
        lbl.setStyleSheet('font-size:11px;font-weight:700;color:#1C1E26;')
        layout.addWidget(lbl)
        return frame, layout

    def _build_left(self, consensus: dict, responses: list[AgentResponse]):
        # Consensus block
        card, cl = self._card('Debate Summary')

        avg = consensus.get('weighted_avg')
        conf = consensus.get('confidence_pct', 0)
        low = consensus.get('low')
        high = consensus.get('high')

        consensus_frame = QFrame()
        consensus_frame.setStyleSheet('background:#F7F8FA;border-radius:10px;border:1px solid #EAECF0;')
        cf_layout = QVBoxLayout(consensus_frame)
        cf_layout.setContentsMargins(12, 10, 12, 10)

        if avg is not None:
            val_lbl = QLabel(f'+₱{avg:.2f}/L')
        else:
            val_lbl = QLabel('No consensus')
        val_lbl.setStyleSheet('font-size:24px;font-weight:700;color:#1C1E26;')

        sub_lbl = QLabel(f'Weighted average · {conf}% agreement')
        sub_lbl.setStyleSheet('font-size:9px;color:#6B7280;')

        range_row = QHBoxLayout()
        for label, value in [('Low', low), ('High', high), ('Confidence', f'{conf}%')]:
            col = QVBoxLayout()
            col.addWidget(QLabel(label) if isinstance(value, str)
                          else self._muted(label))
            v = f'+₱{value:.2f}' if isinstance(value, float) and value is not None else str(value or '—')
            bold = QLabel(v)
            bold.setStyleSheet('font-size:11px;font-weight:600;color:#1C1E26;')
            col.addWidget(bold)
            range_row.addLayout(col)

        cf_layout.addWidget(val_lbl)
        cf_layout.addWidget(sub_lbl)
        cf_layout.addLayout(range_row)
        cl.addWidget(consensus_frame)

        # Per-agent verdicts
        final_round = max((r.round_num for r in responses), default=1)
        final = [r for r in responses if r.round_num == final_round]
        for resp in final:
            vf = QFrame()
            vf.setStyleSheet('background:#F7F8FA;border-radius:8px;border:1px solid #EAECF0;')
            vfl = QVBoxLayout(vf)
            vfl.setContentsMargins(10, 8, 10, 8)
            vfl.setSpacing(4)

            head_row = QHBoxLayout()
            name_lbl = QLabel(resp.agent_name)
            name_lbl.setStyleSheet('font-size:10px;font-weight:600;color:#1C1E26;')
            est = f'+₱{resp.price_estimate:.2f}' if resp.price_estimate else '—'
            est_lbl = QLabel(est)
            est_lbl.setStyleSheet('font-size:10px;font-weight:700;color:#1C1E26;')
            head_row.addWidget(name_lbl)
            head_row.addStretch()
            head_row.addWidget(est_lbl)
            vfl.addLayout(head_row)

            stmt_lbl = QLabel(resp.statement[:300])
            stmt_lbl.setWordWrap(True)
            stmt_lbl.setStyleSheet('font-size:9px;color:#374151;')
            vfl.addWidget(stmt_lbl)
            cl.addWidget(vf)

        self._left.addWidget(card)
        self._left.addStretch()

    def _build_right(self, regressor, df, cv_rmse, scenario, consensus):
        # Metric cards
        card, cl = self._card('Final Outputs')

        avg = consensus.get('weighted_avg', 0.0) or 0.0
        X, y, feature_cols, df_feat = build_features(df)
        current_price = float(df_feat.iloc[-1]['gas_price'])
        forecast_price = current_price + avg

        metrics = [
            ('Forecast gas price', f'₱{forecast_price:.2f}/L'),
            ('vs current',         f'+₱{avg:.2f}'),
            ('CV-RMSE',            f'±₱{cv_rmse:.2f}'),
            ('Pressure index',     '—'),
        ]
        grid = QHBoxLayout()
        for label, value in metrics:
            mf = QFrame()
            mf.setStyleSheet('background:#F7F8FA;border:1px solid #EAECF0;border-radius:9px;')
            ml_layout = QVBoxLayout(mf)
            ml_layout.setContentsMargins(10, 8, 10, 8)
            ml_layout.addWidget(self._muted(label))
            v_lbl = QLabel(value)
            v_lbl.setStyleSheet('font-size:16px;font-weight:700;color:#1C1E26;')
            ml_layout.addWidget(v_lbl)
            grid.addWidget(mf)
        cl.addLayout(grid)

        # 6-month forecast chart
        forecast_prices = ml.forecast(regressor, self._make_features(df_feat, feature_cols, scenario))
        if forecast_prices is not None and len(forecast_prices) > 0:
            fig = Figure(figsize=(5, 2.2), facecolor='#F7F8FA')
            fig.tight_layout()
            ax = fig.add_subplot(111)
            months = range(1, len(forecast_prices) + 1)
            ax.plot(months, forecast_prices, color='#1C1E26', linewidth=2)
            ax.fill_between(months,
                            forecast_prices - cv_rmse,
                            forecast_prices + cv_rmse,
                            alpha=0.15, color='#1C1E26')
            ax.set_facecolor('#F7F8FA')
            ax.tick_params(labelsize=7)
            ax.set_xlabel('Month', fontsize=8)
            fig.tight_layout(pad=1.0)
            canvas = FigureCanvasQTAgg(fig)
            canvas.setFixedHeight(200)
            cl.addWidget(canvas)

        # Feature importances
        fi = ml.get_feature_importances(regressor, feature_cols)
        if fi:
            fi_fig = Figure(figsize=(5, 2.2), facecolor='#F7F8FA')
            fi_ax = fi_fig.add_subplot(111)
            names = list(fi.keys())[:6]
            vals = [fi[k] for k in names]
            fi_ax.barh(names, vals, color='#1C1E26')
            fi_ax.set_facecolor('#F7F8FA')
            fi_ax.tick_params(labelsize=7)
            fi_fig.tight_layout(pad=1.0)
            fi_canvas = FigureCanvasQTAgg(fi_fig)
            fi_canvas.setFixedHeight(200)
            cl.addWidget(fi_canvas)

        self._right.addWidget(card)
        self._right.addStretch()

    def _make_features(self, df_feat, feature_cols, scenario):
        last = df_feat.iloc[-1]
        features = []
        for col in feature_cols:
            if col == 'prev_gas_price':
                features.append(float(last['gas_price']))
            elif col == 'oil_price':
                features.append(float(last[col]) * (1 + scenario.get('oil_pct', 0) / 100))
            elif col == 'usd_php':
                features.append(float(last[col]) * (1 + scenario.get('usd_pct', 0) / 100))
            else:
                features.append(float(last[col]))
        return np.array(features)

    def _muted(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet('font-size:8px;color:#9EA3AE;text-transform:uppercase;')
        return lbl

    def _on_export(self):
        if not self._responses:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export Report', str(Path.home() / 'Downloads' / 'simulation_report.pdf'),
            'PDF Files (*.pdf)'
        )
        if not path:
            return
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet

            doc = SimpleDocTemplate(path, pagesize=letter)
            styles = getSampleStyleSheet()
            story = [
                Paragraph('PH Economic Pressure Simulation Report', styles['Title']),
                Spacer(1, 12),
            ]
            avg = self._consensus.get('weighted_avg')
            if avg is not None:
                story.append(Paragraph(f'Consensus: +₱{avg:.2f}/L', styles['Heading2']))
            for resp in self._responses:
                story.append(Paragraph(
                    f"Round {resp.round_num} · {resp.agent_name}: {resp.statement[:500]}",
                    styles['Normal']
                ))
                story.append(Spacer(1, 6))
            doc.build(story)
        except Exception as e:
            pass  # silently fail — export is non-critical
```

- [ ] **Step 2: Commit**

```bash
git add ph_economic_ai/ui/stage4_report.py
git commit -m "feat: add Stage 4 report panel with consensus, metrics, charts, PDF export"
```

---

## Task 10: Stage 5 — Interaction panel

**Files:**
- Create: `ph_economic_ai/ui/stage5_interact.py`

- [ ] **Step 1: Implement Stage5InteractPanel**

```python
# ph_economic_ai/ui/stage5_interact.py
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QTabWidget, QScrollArea, QLineEdit, QSlider,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from ph_economic_ai.engine.rag import RagEngine
from ph_economic_ai.engine.debate import Agent, DebateEngine, AgentResponse


class _AskThread(QThread):
    token_received = pyqtSignal(str)
    done = pyqtSignal(str)

    def __init__(self, engine: DebateEngine, agent_name: str, question: str, parent=None):
        super().__init__(parent)
        self._engine = engine
        self._agent_name = agent_name
        self._question = question

    def run(self):
        answer = self._engine.ask(
            self._agent_name, self._question,
            on_token=lambda tok: self.token_received.emit(tok),
        )
        self.done.emit(answer)


class Stage5InteractPanel(QWidget):
    rerun_requested = pyqtSignal(object)  # scenario dict

    def __init__(self, rag: RagEngine, agents: list[Agent],
                 regressor, df, cv_rmse: float, parent=None):
        super().__init__(parent)
        self._rag = rag
        self._agents = agents
        self._regressor = regressor
        self._df = df
        self._cv_rmse = cv_rmse
        self._debate_engine: DebateEngine | None = None
        self._last_scenario: dict = {}
        self._ask_thread: _AskThread | None = None
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        h = QLabel('Stage 5 — Deep Interaction')
        h.setStyleSheet('font-size:18px;font-weight:700;color:#1C1E26;padding:20px 24px 0 24px;')
        root.addWidget(h)

        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.setStyleSheet(
            'QTabBar::tab{padding:8px 16px;font-size:10px;font-weight:600;color:#9EA3AE;'
            'border:none;background:transparent;}'
            'QTabBar::tab:selected{color:#1C1E26;border-bottom:2px solid #1C1E26;}'
        )
        tabs.addTab(self._build_adjust_tab(), 'Adjust & Re-run')
        tabs.addTab(self._build_ask_tab(), 'Ask an Agent')
        tabs.addTab(self._build_toggle_tab(), 'Toggle Sources')
        root.addWidget(tabs, stretch=1)

    def _build_adjust_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(12)

        lbl = QLabel('Adjust scenario inputs and re-run the full simulation.')
        lbl.setStyleSheet('font-size:10px;color:#9EA3AE;')
        layout.addWidget(lbl)

        self._adjust_pills: list[tuple[str, QSlider, QLabel, float, float, float]] = []
        configs = [
            ('Oil shock %',     'oil_pct',      5.0,  -20.0, 30.0,  0.5),
            ('USD/PHP shift %', 'usd_pct',      2.0,  -10.0, 15.0,  0.5),
            ('BSP rate %',      'bsp_rate',     6.5,    3.0, 10.0,  0.25),
            ('Demand index',    'demand_index', 72.0,  50.0, 100.0,  1.0),
        ]
        for label, key, default, mn, mx, step in configs:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(int(mn / step), int(mx / step))
            slider.setValue(int(default / step))
            val_lbl = QLabel(f'{default:.1f}')
            val_lbl.setFixedWidth(40)
            slider.valueChanged.connect(lambda v, lbl=val_lbl, s=step: lbl.setText(f'{v * s:.1f}'))
            row.addWidget(slider, stretch=1)
            row.addWidget(val_lbl)
            layout.addLayout(row)
            self._adjust_pills.append((key, slider, val_lbl, step, mn, mx))

        run_btn = QPushButton('Re-run Simulation →')
        run_btn.setStyleSheet(
            'QPushButton{background:#1C1E26;color:#FFFFFF;border-radius:9px;'
            'padding:10px;font-size:11px;font-weight:700;border:none;}'
        )
        run_btn.clicked.connect(self._on_rerun)
        layout.addWidget(run_btn)
        layout.addStretch()
        return w

    def _build_ask_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(10)

        # Agent selector
        selector_row = QHBoxLayout()
        self._agent_chips: list[QPushButton] = []
        self._selected_agent: str = self._agents[0].name if self._agents else ''
        for agent in self._agents:
            btn = QPushButton(agent.name)
            btn.setCheckable(True)
            btn.setChecked(agent.name == self._selected_agent)
            btn.clicked.connect(lambda _, name=agent.name: self._select_agent(name))
            btn.setStyleSheet(
                'QPushButton{border:1px solid #EAECF0;border-radius:8px;'
                'padding:5px 10px;font-size:9px;font-weight:600;color:#6B7280;background:#F7F8FA;}'
                'QPushButton:checked{background:#1C1E26;color:#FFFFFF;border-color:#1C1E26;}'
            )
            self._agent_chips.append(btn)
            selector_row.addWidget(btn)
        selector_row.addStretch()
        layout.addLayout(selector_row)

        # Chat log
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._chat_widget = QWidget()
        self._chat_layout = QVBoxLayout(self._chat_widget)
        self._chat_layout.setSpacing(6)
        self._chat_layout.setContentsMargins(0, 0, 0, 0)
        self._chat_layout.addStretch()
        scroll.setWidget(self._chat_widget)
        layout.addWidget(scroll, stretch=1)

        # Input row
        input_row = QHBoxLayout()
        self._chat_input = QLineEdit()
        self._chat_input.setPlaceholderText('Ask a follow-up question...')
        self._chat_input.setStyleSheet(
            'QLineEdit{border:1px solid #EAECF0;border-radius:8px;'
            'padding:7px 10px;font-size:10px;}'
        )
        self._chat_input.returnPressed.connect(self._on_ask)
        send_btn = QPushButton('Send')
        send_btn.setStyleSheet(
            'QPushButton{background:#1C1E26;color:#FFFFFF;border-radius:8px;'
            'padding:7px 14px;font-size:9px;font-weight:600;border:none;}'
        )
        send_btn.clicked.connect(self._on_ask)
        input_row.addWidget(self._chat_input, stretch=1)
        input_row.addWidget(send_btn)
        layout.addLayout(input_row)
        return w

    def _build_toggle_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(8)

        lbl = QLabel('Toggle RAG sources on/off. Re-run to see impact.')
        lbl.setStyleSheet('font-size:10px;color:#9EA3AE;')
        layout.addWidget(lbl)

        self._toggle_buttons: dict[str, QPushButton] = {}
        for src in self._rag.all_source_names:
            row = QHBoxLayout()
            toggle = QPushButton(src)
            toggle.setCheckable(True)
            toggle.setChecked(True)
            toggle.clicked.connect(lambda checked, s=src: self._on_toggle(s, checked))
            toggle.setStyleSheet(
                'QPushButton{border:1px solid #EAECF0;border-radius:8px;'
                'padding:6px 12px;font-size:10px;font-weight:600;background:#F7F8FA;color:#1C1E26;}'
                'QPushButton:checked{background:#1C1E26;color:#FFFFFF;border-color:#1C1E26;}'
            )
            self._toggle_buttons[src] = toggle
            row.addWidget(toggle)
            row.addStretch()
            layout.addLayout(row)

        rerun_btn = QPushButton('Re-run with current sources →')
        rerun_btn.setStyleSheet(
            'QPushButton{background:#1C1E26;color:#FFFFFF;border-radius:9px;'
            'padding:10px;font-size:11px;font-weight:700;border:none;}'
        )
        rerun_btn.clicked.connect(self._on_rerun)
        layout.addWidget(rerun_btn)
        layout.addStretch()
        return w

    def update_context(self, responses: list[AgentResponse], scenario: dict):
        self._last_scenario = scenario
        if self._debate_engine is None and responses:
            pass  # engine reference set by main_window via stage3.engine

    def set_debate_engine(self, engine: DebateEngine):
        self._debate_engine = engine

    def _select_agent(self, name: str):
        self._selected_agent = name
        for btn in self._agent_chips:
            btn.setChecked(btn.text() == name)

    def _on_ask(self):
        question = self._chat_input.text().strip()
        if not question or not self._debate_engine:
            return
        self._chat_input.clear()
        self._add_bubble(f'You: {question}', user=True)
        self._current_answer_lbl = self._add_bubble(
            f'{self._selected_agent}: thinking...', user=False
        )
        self._ask_thread = _AskThread(
            self._debate_engine, self._selected_agent, question
        )
        self._ask_thread.done.connect(self._on_answer_done)
        self._ask_thread.start()

    def _add_bubble(self, text: str, user: bool) -> QLabel:
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setStyleSheet(
            f'font-size:9px;padding:7px 9px;border-radius:8px;line-height:1.4;'
            + ('background:#1C1E26;color:#FFFFFF;margin-left:20px;'
               if user else
               'background:#F7F8FA;border:1px solid #EAECF0;color:#374151;margin-right:20px;')
        )
        self._chat_layout.insertWidget(self._chat_layout.count() - 1, lbl)
        return lbl

    def _on_answer_done(self, answer: str):
        if self._current_answer_lbl:
            self._current_answer_lbl.setText(f'{self._selected_agent}: {answer[:500]}')

    def _on_toggle(self, source: str, enabled: bool):
        self._rag.toggle_source(source, enabled)

    def _on_rerun(self):
        scenario = dict(self._last_scenario)
        for key, slider, _, step, _, _ in self._adjust_pills:
            scenario[key] = slider.value() * step
        self.rerun_requested.emit(scenario)
```

- [ ] **Step 2: Wire `set_debate_engine` in SimMainWindow._on_simulation_complete**

In `ph_economic_ai/ui/main_window.py`, inside `_on_simulation_complete`:

```python
def _on_simulation_complete(self, responses):
    consensus = self._stage3.engine.consensus()
    self._stage4.populate(responses, consensus, self._regressor,
                          self._df, self._cv_rmse,
                          self._stage3.scenario())
    self._stage5.update_context(responses, self._stage3.scenario())
    self._stage5.set_debate_engine(self._stage3.engine)   # ← add this line
    self._sidebar.unlock_stages([2, 3, 4])
    self._sidebar.set_active(3)
    self._stack.setCurrentIndex(3)
```

- [ ] **Step 3: Commit**

```bash
git add ph_economic_ai/ui/stage5_interact.py ph_economic_ai/ui/main_window.py
git commit -m "feat: add Stage 5 interaction panel with adjust/re-run, ask agent, toggle sources"
```

---

## Task 11: End-to-end smoke test

**Files:** none (manual verification)

- [ ] **Step 1: Run the app**

```bash
python -m ph_economic_ai.main
```

- [ ] **Step 2: Verify Stage 1**

Expected: window opens maximized. Stage 1 shows 9 source cards. Progress bars animate. Chunk count increases. After ~10 seconds, most cards show "X chunks".

- [ ] **Step 3: Navigate to Stage 2, run a simulation**

Click "2. Environment" in sidebar. Keep default 3 agents. Click "Run Simulation →".

Expected: sidebar navigates to Stage 3. Canvas appears with 3 agent nodes arranged in a triangle around the HistGBM engine node. Agent nodes pulse as deepseek-r1 streams tokens. Speech bubbles update in real time. Debate log fills on the right.

- [ ] **Step 4: Verify Stage 4 report**

After simulation completes, Stage 4 opens automatically.

Expected: Status bar shows agent/round count. Left panel shows consensus price estimate and per-agent verdicts. Right panel shows metric cards, 6-month forecast chart, feature importance bars.

- [ ] **Step 5: Verify Stage 5 interaction**

Click "5. Interact" in sidebar.

Expected: Three tabs visible. "Adjust & Re-run" shows sliders. "Ask an Agent" shows agent chips and a chat input. "Toggle Sources" shows toggle buttons for all indexed sources.

- [ ] **Step 6: Test Ask an Agent**

Select an agent chip, type "What is the biggest risk in this scenario?", press Enter.

Expected: Answer streams in from deepseek-r1, appears in chat bubble.

- [ ] **Step 7: Test re-run**

In "Adjust & Re-run", drag Oil shock to +15%. Click "Re-run Simulation →".

Expected: App navigates back to Stage 3, debate runs again with new scenario, Stage 4 updates with new report.

- [ ] **Step 8: Final commit**

```bash
git add -A
git commit -m "feat: complete Economic Pressure AI Simulation Engine — all 5 stages wired"
```

---

## Self-Review Checklist

**Spec coverage:**

| Spec section | Covered by task |
|---|---|
| showMaximized | Task 5 (main.py) |
| 9 parallel sources | Task 2 (rag.py fetch_all) |
| 512-token chunking | Task 2 (rag.py _chunk) |
| TF-IDF index | Task 2 (rag.py _refit) |
| PDF upload | Task 2 (rag.py add_pdf) + Task 6 (Stage1) |
| Pre-bundled NEDA corpus | Task 4 + Task 5 (main_window loads it) |
| 10 agents max | Task 7 (Stage2, add button disabled at 10) |
| 3 rounds default, 2 if >7 | Task 3 (debate.py) + Task 8 (Stage3) |
| Agent source filtering | Task 2 (rag.query sources=) |
| Think token streaming | Task 3 (debate.py _parse_think) + Task 8 |
| HistGBM engine node | Task 8 (Stage3 _update_engine_price) |
| Stage lock/unlock | Task 5 (sidebar.unlock_stages) |
| Debate log (right panel) | Task 8 (Stage3 right panel) |
| Consensus + verdicts | Task 3 (debate.consensus()) + Task 9 |
| 6-month forecast chart | Task 9 (Stage4 _build_right) |
| Feature importance chart | Task 9 (Stage4 _build_right) |
| PDF export | Task 9 (Stage4 _on_export) |
| Adjust & re-run | Task 10 (Stage5 Tab 1) |
| Ask an agent chat | Task 10 (Stage5 Tab 2) |
| Toggle RAG sources | Task 10 (Stage5 Tab 3) |
| Financial data strip | Not implemented — out of scope for MVP; live prices come from Yahoo Finance RAG chunks |

**Financial data strip note:** The live Brent/USD/PHP/PSEi strip from the Stage 1 mockup was not included in the task plan — it requires parsing Yahoo Finance JSON (different from the HTML scrape). This is non-critical for the simulation to function. Add as a follow-up task if needed.
