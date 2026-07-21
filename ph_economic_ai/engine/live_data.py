"""Live Philippine economic data fetcher.

Fetches real-time market prices (Yahoo Finance JSON API) and official PH
government news headlines (Google News RSS) in parallel, then formats them
as a DATA BRIEF string injected at the top of every agent prompt.

Also provides BSP inflation alert logic and the CausalChainThread that
synthesizes a cross-sector causal chain from the three sector verdicts.
"""
from __future__ import annotations

import logging
import html
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Optional

import requests
from PyQt6.QtCore import QThread, pyqtSignal

from ph_economic_ai.engine import llm

# ── HTTP headers ──────────────────────────────────────────────────────────────
_JSON_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    'Accept': 'application/json',
}
_RSS_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    'Accept': 'text/xml,application/xml',
}

# BSP inflation target band (2-4%)
BSP_TARGET_LOW  = 2.0
BSP_TARGET_HIGH = 4.0

# Current CPI baseline from PSA (updated monthly in debate.py constants)
_CURRENT_CPI_PCT = 3.8   # PSA April 2026 headline CPI

# CPI basket weights (PSA 2018 base) used for pass-through math
_FUEL_BASKET_WEIGHT  = 0.089   # Transport fuel share
_FOOD_BASKET_WEIGHT  = 0.388   # Food & non-alcoholic beverages
_ELEC_BASKET_WEIGHT  = 0.032   # Electricity share

# Pass-through coefficients (derived from BSP econometric models)
# ₱1/L fuel change → ppt CPI impact
_FUEL_PASSTHROUGH_PER_PHP   = 0.19
# 1% monthly food price change → ppt CPI impact
_FOOD_PASSTHROUGH_PER_PCT   = 0.388
# ₱0.10/kWh electricity change → ppt CPI impact
_ELEC_PASSTHROUGH_PER_10SEN = 0.072


# ── Yahoo Finance JSON fetcher ────────────────────────────────────────────────

def _yahoo_price(symbol: str, timeout: int = 8, silent: bool = False) -> Optional[float]:
    url = (f'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}'
           f'?range=1d&interval=1d')
    try:
        r = requests.get(url, headers=_JSON_HEADERS, timeout=timeout)
        r.raise_for_status()
        meta = r.json()['chart']['result'][0]['meta']
        return meta.get('regularMarketPrice') or meta.get('previousClose')
    except Exception as e:
        if not silent:
            logging.warning('live_data: yahoo %s failed: %s', symbol, e)
        return None


def _yahoo_history(symbol: str, days: int = 5) -> list[tuple[str, float]]:
    """Return [(date, close), ...] for the last N trading days."""
    url = (f'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}'
           f'?range=1mo&interval=1d')
    try:
        r = requests.get(url, headers=_JSON_HEADERS, timeout=8)
        r.raise_for_status()
        data  = r.json()['chart']['result'][0]
        ts    = data.get('timestamp', [])
        closes = data['indicators']['quote'][0].get('close', [])
        pairs  = [(datetime.fromtimestamp(t, tz=timezone.utc).strftime('%b %d'), c)
                  for t, c in zip(ts, closes) if c is not None]
        return pairs[-days:]
    except Exception:
        return []


# ── RSS headline fetcher ──────────────────────────────────────────────────────

def _rss_headlines(url: str, limit: int = 3) -> list[str]:
    def _clean_text(value: str) -> str:
        value = html.unescape(value or '')
        return re.sub(r'<[^>]+>', '', value).strip()

    def _repair_named_entities(value: str) -> str:
        # XML knows only amp/lt/gt/quot/apos plus numeric entities. Google News
        # can include HTML-only names such as &nbsp; or &mdash; in titles.
        xml_entities = {'amp', 'lt', 'gt', 'quot', 'apos'}

        def repl(match):
            name = match.group(1)
            if name in xml_entities or name.startswith('#'):
                return match.group(0)
            return html.unescape(match.group(0))

        return re.sub(r'&([A-Za-z][A-Za-z0-9]+|#\d+|#x[0-9A-Fa-f]+);', repl, value)

    def _regex_items(value: str) -> list[str]:
        out = []
        for block in re.findall(r'<item\b.*?</item>', value, flags=re.IGNORECASE | re.DOTALL):
            title_m = re.search(r'<title>(.*?)</title>', block, flags=re.IGNORECASE | re.DOTALL)
            desc_m = re.search(r'<description>(.*?)</description>', block, flags=re.IGNORECASE | re.DOTALL)
            title = _clean_text(title_m.group(1) if title_m else '')
            desc = _clean_text(desc_m.group(1) if desc_m else '')
            line = f'{title}. {desc}'.strip('. ')
            if line:
                out.append(line[:200])
            if len(out) >= limit:
                break
        return out

    try:
        r = requests.get(url, headers=_RSS_HEADERS, timeout=8)
        r.raise_for_status()

        text = r.content.decode(r.apparent_encoding or 'utf-8', errors='replace')
        text = _repair_named_entities(text)
        try:
            root = ET.fromstring(text)
        except ET.ParseError:
            fallback = _regex_items(text)
            if fallback:
                return fallback
            raise

        items = root.findall('.//item')[:limit]
        out   = []
        for item in items:
            title = _clean_text(item.findtext('title') or '')
            desc  = _clean_text(item.findtext('description') or '')
            line  = f'{title}. {desc}'.strip('. ')
            if line:
                out.append(line[:200])
        return out
    except Exception as e:
        logging.warning('live_data: rss failed %s: %s', url, e)
        return []


# ── Named fetchers ────────────────────────────────────────────────────────────

def fetch_brent() -> Optional[float]:
    return _yahoo_price('BZ=F')

def fetch_usd_php() -> Optional[float]:
    return _yahoo_price('PHP=X')

def fetch_psei() -> Optional[float]:
    # Yahoo's chart API does not reliably expose PSEi. It is optional context,
    # so fail quietly instead of warning on every app launch.
    return _yahoo_price('^PSEI', silent=True)

def fetch_wti() -> Optional[float]:
    return _yahoo_price('CL=F')

def fetch_bsp_headlines() -> list[str]:
    url = ('https://news.google.com/rss/search?q=BSP+policy+rate+Philippines'
           '+inflation+2026&hl=en&gl=PH&ceid=PH:en')
    return _rss_headlines(url, 3)

def fetch_doe_headlines() -> list[str]:
    url = ('https://news.google.com/rss/search?q=DOE+oil+price+bulletin'
           '+Philippines+per+liter+2026&hl=en&gl=PH&ceid=PH:en')
    return _rss_headlines(url, 3)

def fetch_psa_headlines() -> list[str]:
    url = ('https://news.google.com/rss/search?q=PSA+CPI+inflation'
           '+Philippines+2026&hl=en&gl=PH&ceid=PH:en')
    return _rss_headlines(url, 3)

def fetch_nfa_headlines() -> list[str]:
    url = ('https://news.google.com/rss/search?q=NFA+rice+buffer+stock'
           '+Philippines+2026&hl=en&gl=PH&ceid=PH:en')
    return _rss_headlines(url, 2)


def derive_scenario_from_brief(brief: Optional['LiveDataBrief'],
                               current_price: Optional[float] = None) -> dict:
    """Derive a scenario dict from observed live data with sensible defaults.

    - oil_pct          : % change of Brent vs ~5-day-ago price (from brief.brent_hist)
    - usd_pct          : % change of USD/PHP vs ~5-day-ago rate (from brief.fx_hist)
    - bsp_rate         : 6.5 (BSP overnight RRP — only changes 4–6×/year, default reasonable)
    - demand_index     : derived from Manila max-temp (hotter day → higher consumption proxy)
                         falls back to 72 if weather missing
    - current_price    : passed in (live retail) or 98.82 fallback
    """
    def _pct_change(series: list) -> float:
        if not series or len(series) < 2:
            return 0.0
        try:
            first = series[0][1] if isinstance(series[0], (list, tuple)) else series[0]
            last  = series[-1][1] if isinstance(series[-1], (list, tuple)) else series[-1]
            if first:
                return round((last - first) / first * 100, 2)
        except Exception:
            pass
        return 0.0

    oil_pct  = 0.0
    usd_pct  = 0.0
    demand_idx = 72.0

    if brief is not None:
        oil_pct = _pct_change(brief.brent_hist or [])
        usd_pct = _pct_change(brief.fx_hist or [])
        w = brief.weather_manila or {}
        tmax = w.get('today_max')
        if isinstance(tmax, (int, float)):
            # Map 25°C → 60 (cool), 30°C → 72, 35°C → 84, clamped 50–95
            demand_idx = max(50.0, min(95.0, 60.0 + (tmax - 25.0) * 2.4))

    return {
        'oil_pct':       float(oil_pct),
        'usd_pct':       float(usd_pct),
        'bsp_rate':      6.5,
        'demand_index':  round(demand_idx, 1),
        'current_price': float(current_price) if current_price is not None else 98.82,
    }


def fetch_open_meteo_manila(timeout: int = 8) -> Optional[dict]:
    """Fetch a compact 7-day Manila weather summary from Open-Meteo.

    Returns a dict with: now_temp_c, today_min, today_max, today_rain_mm,
    week_rain_mm, week_wet_days. Returns None on any network/parse failure.
    """
    url = ('https://api.open-meteo.com/v1/forecast'
           '?latitude=14.6042&longitude=120.9822'
           '&current=temperature_2m'
           '&daily=temperature_2m_max,temperature_2m_min,precipitation_sum'
           '&timezone=Asia/Manila&forecast_days=7')
    try:
        r = requests.get(url, headers=_JSON_HEADERS, timeout=timeout)
        r.raise_for_status()
        d = r.json()
        daily = d.get('daily', {}) or {}
        rain = daily.get('precipitation_sum', []) or []
        tmax = daily.get('temperature_2m_max', []) or []
        tmin = daily.get('temperature_2m_min', []) or []
        cur  = d.get('current', {}) or {}
        return {
            'now_temp_c':    cur.get('temperature_2m'),
            'today_max':     tmax[0] if tmax else None,
            'today_min':     tmin[0] if tmin else None,
            'today_rain_mm': rain[0] if rain else None,
            'week_rain_mm':  sum(rain) if rain else None,
            'week_wet_days': sum(1 for r in rain if r > 5) if rain else None,
        }
    except Exception as e:
        logging.warning('live_data: open-meteo failed: %s', e)
        return None


# ── LiveDataBrief ─────────────────────────────────────────────────────────────

class LiveDataBrief:
    """Fetches and caches all live PH economic data for one simulation run.

    Usage:
        brief = LiveDataBrief().fetch()
        prompt_injection = brief.as_prompt_block(scenario_dict)
    """

    def __init__(self):
        self.brent:     Optional[float] = None
        self.wti:       Optional[float] = None
        self.usd_php:   Optional[float] = None
        self.psei:      Optional[float] = None
        self.brent_hist: list[tuple[str, float]] = []
        self.fx_hist:    list[tuple[str, float]] = []

        self.bsp_news:  list[str] = []
        self.doe_news:  list[str] = []
        self.psa_news:  list[str] = []
        self.nfa_news:  list[str] = []

        self.weather_manila: Optional[dict] = None

        self.fetched_at: str = ''
        self._ok: bool = False

    def fetch(self) -> 'LiveDataBrief':
        """Parallel fetch. Safe to call from a background thread."""
        with ThreadPoolExecutor(max_workers=11) as pool:
            f = {
                'brent':          pool.submit(fetch_brent),
                'wti':            pool.submit(fetch_wti),
                'usd_php':        pool.submit(fetch_usd_php),
                'psei':           pool.submit(fetch_psei),
                'brent_hist':     pool.submit(_yahoo_history, 'BZ=F', 5),
                'fx_hist':        pool.submit(_yahoo_history, 'PHP=X', 5),
                'bsp_news':       pool.submit(fetch_bsp_headlines),
                'doe_news':       pool.submit(fetch_doe_headlines),
                'psa_news':       pool.submit(fetch_psa_headlines),
                'nfa_news':       pool.submit(fetch_nfa_headlines),
                'weather_manila': pool.submit(fetch_open_meteo_manila),
            }
            self.brent          = f['brent'].result()
            self.wti            = f['wti'].result()
            self.usd_php        = f['usd_php'].result()
            self.psei           = f['psei'].result()
            self.brent_hist     = f['brent_hist'].result()
            self.fx_hist        = f['fx_hist'].result()
            self.bsp_news       = f['bsp_news'].result()
            self.doe_news       = f['doe_news'].result()
            self.psa_news       = f['psa_news'].result()
            self.nfa_news       = f['nfa_news'].result()
            self.weather_manila = f['weather_manila'].result()

        self.fetched_at = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
        self._ok = True
        return self

    # ── Prompt injection ──────────────────────────────────────────────────────

    def as_prompt_block(self, scenario: dict) -> str:
        """Return a DATA BRIEF string to prepend to every agent prompt.

        Agents receive actual live market numbers and official PH government
        news so they can cite specific data instead of making vague claims.
        """
        lines: list[str] = []
        lines.append('╔══════════════════════════════════════════════════╗')
        lines.append('║     LIVE PHILIPPINE ECONOMIC DATA BRIEF          ║')
        lines.append(f'║     {self.fetched_at:<44}║')
        lines.append('╚══════════════════════════════════════════════════╝')
        lines.append('')

        lines.append('■ INTERNATIONAL COMMODITY PRICES')
        if self.brent is not None:
            lines.append(f'  Brent Crude (ICE):    ${self.brent:.2f}/bbl')
        if self.wti is not None:
            lines.append(f'  WTI Crude (NYMEX):    ${self.wti:.2f}/bbl')
        if self.brent_hist:
            hist_str = '  '.join(f'{d}: ${p:.1f}' for d, p in self.brent_hist[-3:])
            lines.append(f'  Brent 3-day trend:    {hist_str}')

        lines.append('')
        lines.append('■ PHILIPPINE FINANCIAL MARKETS')
        if self.usd_php is not None:
            lines.append(f'  USD/PHP Exchange Rate: ₱{self.usd_php:.4f}')
        if self.fx_hist:
            fx_str = '  '.join(f'{d}: ₱{p:.2f}' for d, p in self.fx_hist[-3:])
            lines.append(f'  FX 3-day trend:        {fx_str}')
        if self.psei is not None:
            lines.append(f'  PSEi (PH Stocks):      {self.psei:,.2f} pts')

        if self.weather_manila:
            w = self.weather_manila
            lines.append('')
            lines.append('■ MANILA WEATHER (Open-Meteo, 7-day)')
            if w.get('now_temp_c') is not None:
                lines.append(f'  Now:                   {w["now_temp_c"]:.1f}°C')
            if w.get('today_min') is not None and w.get('today_max') is not None:
                lines.append(f'  Today range:           {w["today_min"]:.1f}–{w["today_max"]:.1f}°C')
            if w.get('today_rain_mm') is not None:
                lines.append(f'  Today rainfall:        {w["today_rain_mm"]:.1f}mm')
            if w.get('week_rain_mm') is not None and w.get('week_wet_days') is not None:
                lines.append(f'  7-day rainfall total:  {w["week_rain_mm"]:.1f}mm '
                             f'({w["week_wet_days"]} wet days)')

        lines.append('')
        lines.append('■ SIMULATION SCENARIO')
        lines.append(f'  Oil price shock:      {scenario.get("oil_pct", 0):+.1f}%')
        lines.append(f'  USD/PHP shift:        {scenario.get("usd_pct", 0):+.1f}%')
        lines.append(f'  BSP overnight rate:   {scenario.get("bsp_rate", 6.5):.2f}%')
        lines.append(f'  Demand index:         {scenario.get("demand_index", 72):.0f}/100')
        lines.append(f'  Current pump price:   ₱{scenario.get("current_price", 98.82):.2f}/L')

        if self.bsp_news:
            lines.append('')
            lines.append('■ BSP / MONETARY POLICY (latest headlines)')
            for h in self.bsp_news[:2]:
                lines.append(f'  • {h}')

        if self.doe_news:
            lines.append('')
            lines.append('■ DOE FUEL PRICE SIGNALS (latest headlines)')
            for h in self.doe_news[:2]:
                lines.append(f'  • {h}')

        if self.psa_news:
            lines.append('')
            lines.append('■ PSA / INFLATION SIGNALS (latest headlines)')
            for h in self.psa_news[:2]:
                lines.append(f'  • {h}')

        if self.nfa_news:
            lines.append('')
            lines.append('■ NFA / FOOD SECURITY SIGNALS')
            for h in self.nfa_news[:2]:
                lines.append(f'  • {h}')

        lines.append('')
        lines.append('You MUST cite specific numbers from the DATA BRIEF above when')
        lines.append('making your estimate. Vague claims unsupported by data will be')
        lines.append('penalised in the scoring round.')
        lines.append('═' * 50)
        return '\n'.join(lines)

    # ── BSP alert ─────────────────────────────────────────────────────────────

    @staticmethod
    def check_bsp_alert(
        gas_php_per_l:    Optional[float],
        food_pct:         Optional[float],
        elec_php_per_kwh: Optional[float],
        current_cpi:      float = _CURRENT_CPI_PCT,
    ) -> dict:
        """Estimate whether sector verdicts would breach BSP 2-4% CPI target.

        Uses PSA basket weights and BSP pass-through coefficients:
          Fuel:  ₱1/L  → ~0.19 ppt CPI  (transport fuel basket share 8.9%)
          Food:  1%/mo → ~0.39 ppt CPI  (food basket share 38.8%)
          Elec:  ₱0.10/kWh → ~0.07 ppt CPI (electricity basket share 3.2%)
        """
        contrib: dict[str, float] = {}
        total   = 0.0

        if gas_php_per_l is not None:
            c = gas_php_per_l * _FUEL_PASSTHROUGH_PER_PHP
            contrib['fuel']  = round(c, 3)
            total += c

        if food_pct is not None:
            c = food_pct * _FOOD_PASSTHROUGH_PER_PCT
            contrib['food']  = round(c, 3)
            total += c

        if elec_php_per_kwh is not None:
            c = (elec_php_per_kwh / 0.10) * _ELEC_PASSTHROUGH_PER_10SEN
            contrib['electricity'] = round(c, 3)
            total += c

        projected = current_cpi + total
        return {
            'current_cpi':        round(current_cpi, 2),
            'sector_cpi_impact':  round(total, 3),
            'projected_cpi':      round(projected, 2),
            'bsp_target_low':     BSP_TARGET_LOW,
            'bsp_target_high':    BSP_TARGET_HIGH,
            'breaches_upper':     projected > BSP_TARGET_HIGH,
            'within_target':      BSP_TARGET_LOW <= projected <= BSP_TARGET_HIGH,
            'breakdown':          contrib,
            'severity': (
                'CRITICAL' if projected > BSP_TARGET_HIGH + 1.0 else
                'ALERT'    if projected > BSP_TARGET_HIGH else
                'WATCH'    if projected > BSP_TARGET_HIGH - 0.3 else
                'STABLE'
            ),
        }


# ── Live Brief Fetch Thread ───────────────────────────────────────────────────

class LiveBriefThread(QThread):
    """Fetches LiveDataBrief off the UI thread and emits when ready.

    Connect `ready` signal to a slot that then starts the debate engines,
    guaranteeing agents always receive a populated DATA BRIEF.
    A QTimer fallback in the caller should emit None after a timeout so
    debates still start even if all network calls fail.
    """
    ready = pyqtSignal(object)   # LiveDataBrief | None

    def run(self):
        try:
            brief = LiveDataBrief().fetch()
            self.ready.emit(brief)
        except Exception as e:
            logging.warning('LiveBriefThread failed: %s', e)
            self.ready.emit(None)


# ── Causal Chain Thread ───────────────────────────────────────────────────────

_CHAIN_TIER = llm.DEEP

_CHAIN_SYSTEM = """\
You are a Philippine macroeconomic policy analyst.
Produce a causal chain showing how an economic shock propagates:
trigger → oil market → fuel price → transport → food/electricity → household → BSP policy.

Respond with ONLY valid JSON — no markdown, no explanation, no text outside the JSON.
Use this exact schema:
{"chain": [{"label": "string", "mechanism": "string", "magnitude": "string"}, ...]}

Rules:
- 6 to 8 steps total
- label: node title, 5 words max
- mechanism: one sentence explaining the causal link to the next step
- magnitude: quantified effect using actual numbers from the verdicts (e.g. "+₱1.42/L", "+2.1%", "+0.34 ppt CPI")
- Last step must state projected household CPI impact and BSP policy signal

Example output:
{"chain": [
  {"label": "Oil Price Shock", "mechanism": "OPEC supply cut lifts Brent crude, raising PH import parity cost", "magnitude": "+$8.20/bbl"},
  {"label": "Landed Cost Rise", "mechanism": "Higher crude increases Philippine import parity price for RON91 fuel", "magnitude": "+₱2.10/L"},
  {"label": "Pump Price Hike", "mechanism": "DOE automatic pricing formula passes landed cost increase to consumers", "magnitude": "+₱1.42/L retail"},
  {"label": "Transport Cost", "mechanism": "Higher diesel raises logistics and freight costs across all supply chains", "magnitude": "+3.2% freight"},
  {"label": "Food Price Pressure", "mechanism": "Rising transport costs push up retail food prices through the supply chain", "magnitude": "+1.8% food CPI"},
  {"label": "Electricity Surcharge", "mechanism": "Generation fuel cost activates ERC pass-through surcharge on meralco bills", "magnitude": "+₱0.42/kWh"},
  {"label": "Household Squeeze", "mechanism": "Combined fuel, food, and electricity increases erode real household income", "magnitude": "+0.8 ppt CPI"},
  {"label": "BSP Policy Signal", "mechanism": "Projected CPI above 4% target raises probability of BSP rate hold or hike", "magnitude": "WATCH — hold likely"}
]}
"""

_CHAIN_USER = """\
GAS SECTOR VERDICT:
{gas}

FOOD SECTOR VERDICT:
{food}

ELECTRICITY SECTOR VERDICT:
{elec}

SCENARIO: {scenario}

Using the actual numbers from the verdicts above, produce the causal chain JSON:
"""


class CausalChainStep:
    """One node in the causal chain."""
    __slots__ = ('label', 'mechanism', 'magnitude')

    def __init__(self, label: str, mechanism: str, magnitude: str):
        self.label     = label.strip()
        self.mechanism = mechanism.strip()
        self.magnitude = magnitude.strip()

    def __repr__(self):
        return f'CausalChainStep({self.label!r}, {self.magnitude!r})'


def parse_chain(text: str) -> list[CausalChainStep]:
    """Parse JSON LLM output into CausalChainStep list."""
    import json
    data = json.loads(text)
    return [
        CausalChainStep(
            label=step['label'],
            mechanism=step['mechanism'],
            magnitude=step.get('magnitude', ''),
        )
        for step in data['chain']
        if 'label' in step and 'mechanism' in step
    ]


class CausalChainThread(QThread):
    """Synthesizes cross-sector causal chain from all three sector verdicts."""
    chain_ready = pyqtSignal(list)   # list[CausalChainStep]
    error       = pyqtSignal(str)

    def __init__(
        self,
        gas_verdict:  str,
        food_verdict: str,
        elec_verdict: str,
        scenario:     dict,
        parent=None,
    ):
        super().__init__(parent)
        self._gas      = gas_verdict
        self._food     = food_verdict
        self._elec     = elec_verdict
        self._scenario = scenario

    def run(self):
        try:
            scenario_str = (
                f"Oil shock {self._scenario.get('oil_pct', 0):+.1f}%, "
                f"USD/PHP {self._scenario.get('usd_pct', 0):+.1f}%, "
                f"BSP rate {self._scenario.get('bsp_rate', 6.5):.2f}%, "
                f"demand index {self._scenario.get('demand_index', 72):.0f}"
            )
            user_msg = _CHAIN_USER.format(
                gas=self._gas[:600],
                food=self._food[:600],
                elec=self._elec[:600],
                scenario=scenario_str,
            )
            text = llm.complete(
                [
                    {'role': 'system', 'content': _CHAIN_SYSTEM},
                    {'role': 'user',   'content': user_msg},
                ],
                tier=_CHAIN_TIER,
                json_mode=True,
            )
            steps = parse_chain(text)
            if steps:
                self.chain_ready.emit(steps)
            else:
                self.error.emit('Chain JSON had no steps')
        except Exception as e:
            self.error.emit(f'CausalChainThread: {e}')


# ── Policy Recommendation Thread ──────────────────────────────────────────────

_RECO_TIER = llm.DEEP

_RECO_SYSTEM = """\
You are a senior economic policy adviser to the Philippine government.
Given sector verdicts on fuel, food, and electricity prices, generate exactly 3 \
specific, actionable policy recommendations ordered by urgency (most immediate first).

Philippine policy levers available:
- OPSF: Oil Price Stabilization Fund — DOE releases buffer funds to subsidize pump prices. \
  Each ₱1B release ≈ ₱0.50/L relief for 2-3 weeks.
- NFA: National Food Authority buffer stock release — reduces retail rice/grain prices 5-15%. \
  DA/NFA authorization required.
- BSP RATE: Bangko Sentral policy rate — each 25bps cut risks ₱0.20/L additional landed \
  fuel cost via PHP depreciation; hike has opposite effect. Monetary Board decides.
- EXCISE: Fuel excise tax suspension — saves ₱2.65/L diesel, ₱6.00/L gasoline. \
  Requires Presidential EO (Section 84, NIRC) or Congressional action.
- DSWD: Ayuda/cash transfer top-up to bottom 40% households — ₱5,000–10,000 per family. \
  DSWD/DBM authorization.
- ERC: Electricity Rate Commission deferral of generation fuel surcharge pass-through. \
  ERC en banc order required.
- TARIFF: Presidential EO reducing import tariff on rice/corn/sugar (e.g., 0% rice tariff \
  saves ₱3-5/kg retail). Executive action.
- DTI SRP: Suggested Retail Price enforcement on basic goods basket. DTI administrative order.

Respond ONLY with valid JSON — no markdown, no preamble — matching this exact schema:
{"recommendations": [
  {
    "lever": "lever name from the list above",
    "action": "specific action with peso amounts, basis points, or volume",
    "impact": "projected consumer or market impact with specific numbers",
    "timeline": "implementation timeline and which authority approves it",
    "risk": "main risk or binding constraint to this action"
  }
]}

Use actual numbers extracted from the verdicts. Be specific — avoid vague language.
First recommendation = immediate relief (days). Third = medium-term structural (weeks/months).
"""

_RECO_USER = """\
GAS SECTOR VERDICT:
{gas}

FOOD SECTOR VERDICT:
{food}

ELECTRICITY SECTOR VERDICT:
{elec}

SCENARIO: Oil shock {oil_pct:+.1f}%, USD/PHP {usd_pct:+.1f}%, BSP rate {bsp_rate:.2f}%, \
demand index {demand_index:.0f}.

Generate 3 policy recommendations using the actual numbers from the verdicts above:
"""


class PolicyReco:
    """One policy recommendation from the AI adviser."""
    __slots__ = ('lever', 'action', 'impact', 'timeline', 'risk')

    def __init__(self, lever: str, action: str, impact: str,
                 timeline: str, risk: str):
        self.lever    = lever.strip()
        self.action   = action.strip()
        self.impact   = impact.strip()
        self.timeline = timeline.strip()
        self.risk     = risk.strip()

    def __repr__(self):
        return f'PolicyReco({self.lever!r}, {self.action[:40]!r})'


class PolicyRecoThread(QThread):
    """Generates policy recommendations from all three sector verdicts."""
    reco_ready = pyqtSignal(list)   # list[PolicyReco]
    error      = pyqtSignal(str)

    def __init__(
        self,
        gas_verdict:  str,
        food_verdict: str,
        elec_verdict: str,
        scenario:     dict,
        parent=None,
    ):
        super().__init__(parent)
        self._gas      = gas_verdict
        self._food     = food_verdict
        self._elec     = elec_verdict
        self._scenario = scenario

    def run(self):
        import json
        try:
            user_msg = _RECO_USER.format(
                gas=self._gas[:600],
                food=self._food[:600],
                elec=self._elec[:600],
                oil_pct=self._scenario.get('oil_pct', 0),
                usd_pct=self._scenario.get('usd_pct', 0),
                bsp_rate=self._scenario.get('bsp_rate', 6.5),
                demand_index=self._scenario.get('demand_index', 72),
            )
            data = json.loads(llm.complete(
                [
                    {'role': 'system', 'content': _RECO_SYSTEM},
                    {'role': 'user',   'content': user_msg},
                ],
                tier=_RECO_TIER,
                json_mode=True,
            ))
            recos = [
                PolicyReco(
                    lever=r.get('lever', ''),
                    action=r.get('action', ''),
                    impact=r.get('impact', ''),
                    timeline=r.get('timeline', ''),
                    risk=r.get('risk', ''),
                )
                for r in data.get('recommendations', [])
                if r.get('lever') and r.get('action')
            ]
            if recos:
                self.reco_ready.emit(recos)
            else:
                self.error.emit('PolicyRecoThread: no recommendations in JSON')
        except Exception as e:
            self.error.emit(f'PolicyRecoThread: {e}')
