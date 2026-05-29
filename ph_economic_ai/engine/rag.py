import concurrent.futures
import json
import logging
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer

# Embedding model pulled via ollama for semantic search.
# Pull once with: ollama pull nomic-embed-text
_EMBED_MODEL = 'nomic-embed-text'
_EMBED_WORKERS = 8   # parallel embedding requests
_EMBED_MAX_CHARS = 2000  # truncate chunk text before embedding

_CHUNK_SIZE = 2048    # ~512 tokens at 4 chars/token
_CHUNK_OVERLAP = 256  # ~64 tokens overlap

# Source name → URL for parallel startup fetch
SOURCES: dict[str, str] = {
    'DOE':               'https://www.doe.gov.ph/',
    'BSP':               'https://www.bsp.gov.ph/',
    'BusinessWorld':     'https://www.bworldonline.com/economy/',
    'Reuters':           'https://news.google.com/rss/search?q=philippines+fuel+gasoline+oil&hl=en&gl=PH&ceid=PH:en',
    'Inquirer':          'https://newsinfo.inquirer.net/feed/',
    'ManilaBulletin':    'https://news.google.com/rss/search?q=site:mb.com.ph+gasoline+fuel+price&hl=en&gl=PH&ceid=PH:en',
    'OPEC':              'https://news.google.com/rss/search?q=OPEC+production+oil+crude+2025&hl=en&gl=US&ceid=US:en',
    'YahooFinanceCrude': 'https://query1.finance.yahoo.com/v8/finance/chart/BZ=F?range=1mo&interval=1d',
    'YahooFinanceForex': 'https://query1.finance.yahoo.com/v8/finance/chart/PHP=X?range=1mo&interval=1d',
    # Philippine retail pump price sources
    'DOEBulletin':       'https://news.google.com/rss/search?q=DOE+oil+price+bulletin+Philippines+gasoline+per+liter&hl=en&gl=PH&ceid=PH:en',
    'PHRetailFuel':      'https://news.google.com/rss/search?q=Philippines+pump+price+gasoline+peso+per+liter+this+week&hl=en&gl=PH&ceid=PH:en',
    # PAGASA weather for food sector agent
    'PAGASAWeather':     'https://news.google.com/rss/search?q=PAGASA+weather+rainfall+Philippines+forecast&hl=en&gl=PH&ceid=PH:en',
    # Structured JSON APIs for food / electricity / weather
    'OpenMeteoManila':   'https://api.open-meteo.com/v1/forecast?latitude=14.6042&longitude=120.9822&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code&timezone=Asia/Manila&forecast_days=7',
    'WBPhilFood':        'https://api.worldbank.org/v2/country/PHL/indicator/AG.PRD.FOOD.XD?format=json&date=2015:2025&per_page=30',
    'EIAElectricity':    'https://api.eia.gov/v2/international/data/?api_key={EIA_API_KEY}&data[0]=value&facets[countryRegionId][]=PHL&facets[productId][]=2&frequency=annual&start=2018&end=2024',
    # News-RSS feeds for food / electricity sectors
    'NFARiceRetail':     'https://news.google.com/rss/search?q=NFA+rice+retail+price+Philippines&hl=en&gl=PH&ceid=PH:en',
    'MeralcoCharge':     'https://news.google.com/rss/search?q=Meralco+generation+charge+rate+Philippines&hl=en&gl=PH&ceid=PH:en',
    'WESMSpot':          'https://news.google.com/rss/search?q=WESM+electricity+spot+price+Philippines&hl=en&gl=PH&ceid=PH:en',
}

# Sources that use RSS XML instead of HTML scraping
_RSS_SOURCES = {'Reuters', 'Inquirer', 'ManilaBulletin', 'OPEC', 'DOEBulletin', 'PHRetailFuel',
                'PAGASAWeather', 'NFARiceRetail', 'MeralcoCharge', 'WESMSpot'}

# Sources that return JSON
_JSON_SOURCES = {'YahooFinanceCrude', 'YahooFinanceForex',
                 'OpenMeteoManila', 'WBPhilFood', 'EIAElectricity'}

_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}

_JSON_HEADERS = {
    'User-Agent': _HEADERS['User-Agent'],
    'Accept': 'application/json',
}


@dataclass
class Chunk:
    source: str
    url: str
    timestamp: str
    text: str


# ── JSON parsers — pure functions: dict → searchable text ─────────────────────

def _parse_yahoo(data: dict, source_name: str = '') -> str:
    """Yahoo Finance chart API → price summary + 30-day history."""
    result = data.get('chart', {}).get('result', [{}])[0]
    meta = result.get('meta', {})
    symbol = meta.get('symbol', source_name)
    currency = meta.get('currency', 'USD')
    price = meta.get('regularMarketPrice') or meta.get('previousClose', 0)
    prev = meta.get('previousClose', 0)
    pct = ((price - prev) / prev * 100) if prev else 0
    timestamps = result.get('timestamp', [])
    closes = (result.get('indicators', {})
              .get('quote', [{}])[0]
              .get('close', []))
    history_lines: list[str] = []
    for ts, close in zip(timestamps[-10:], closes[-10:]):
        if close is not None:
            dt = datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d')
            history_lines.append(f"{dt}: {close:.2f} {currency}")
    history = '\n'.join(history_lines)
    return (
        f"{symbol} — current price: {price:.2f} {currency} "
        f"({pct:+.2f}% vs previous close)\n"
        f"30-day history:\n{history}"
    )


def _parse_open_meteo(data: dict, source_name: str = '') -> str:
    """Open-Meteo /v1/forecast → 7-day Manila weather summary."""
    cur = data.get('current', {}) or {}
    daily = data.get('daily', {}) or {}
    hourly = data.get('hourly', {}) or {}
    lines = ['Manila 7-day weather forecast (Open-Meteo).']
    if cur:
        t = cur.get('temperature_2m')
        w = cur.get('wind_speed_10m')
        if t is not None and w is not None:
            lines.append(f"Now: {t:.1f}°C, wind {w} km/h.")
    times = daily.get('time', []) or []
    tmax = daily.get('temperature_2m_max', []) or []
    tmin = daily.get('temperature_2m_min', []) or []
    rain = daily.get('precipitation_sum', []) or []
    code = daily.get('weather_code', []) or []
    for i, t in enumerate(times):
        if i < len(tmin) and i < len(tmax) and i < len(rain):
            wmo = code[i] if i < len(code) else 0
            lines.append(f"{t}: {tmin[i]:.1f}-{tmax[i]:.1f}°C, "
                         f"rain {rain[i]:.1f}mm, wmo_code {wmo}.")
    if rain:
        wet_days = sum(1 for r in rain if r > 5)
        lines.append(f"7-day rainfall total: {sum(rain):.1f}mm. "
                     f"Wet days (>5mm): {wet_days}.")
    rh = hourly.get('relative_humidity_2m', []) or []
    if rh:
        lines.append(f"Humidity range next 7d: {min(rh)}-{max(rh)}%, "
                     f"avg {sum(rh) // len(rh)}%.")
    return '\n'.join(lines)


def _parse_wb_phil_food(data, source_name: str = '') -> str:
    """World Bank Philippines food production index → annual time series.

    Response shape: a top-level list whose second element is the array of rows:
        [ {meta...}, [ {date, value, indicator, ...}, ... ] ]
    """
    if not isinstance(data, list) or len(data) < 2 or not data[1]:
        return ''
    rows = data[1]
    # Sort ascending by year (WB returns descending by default)
    cleaned = sorted(
        ((r.get('date'), r.get('value')) for r in rows if r.get('value') is not None),
        key=lambda t: t[0] or '',
    )
    if not cleaned:
        return ''
    lines = ['Philippine food production index '
             '(World Bank AG.PRD.FOOD.XD, 2014-16 base = 100).']
    for year, value in cleaned:
        lines.append(f"{year}: {value:.2f}")
    first_val = cleaned[0][1]
    last_val = cleaned[-1][1]
    if first_val:
        delta_pct = (last_val - first_val) / first_val * 100
        lines.append(f"Latest ({cleaned[-1][0]}) vs earliest ({cleaned[0][0]}): "
                     f"{delta_pct:+.1f}% change.")
    return '\n'.join(lines)


def _parse_eia(data: dict, source_name: str = '') -> str:
    """EIA v2 international electricity API → Philippine annual electricity.

    Returns generation / consumption / losses time series in BKWH, plus
    earliest-vs-latest growth rates.
    """
    rows = (data.get('response') or {}).get('data') or []
    if not rows:
        return ''
    # Keep only BKWH rows for a single consistent unit
    from collections import defaultdict
    by_year_activity: dict = defaultdict(dict)
    for r in rows:
        if r.get('unit') != 'BKWH':
            continue
        year = r.get('period')
        activity = r.get('activityName')
        value = r.get('value')
        if year is None or activity is None or value is None:
            continue
        try:
            by_year_activity[year][activity] = float(value)
        except (TypeError, ValueError):
            continue
    if not by_year_activity:
        return ''
    lines = ['Philippine electricity (EIA international, annual, units BKWH).']
    for year in sorted(by_year_activity):
        acts = by_year_activity[year]
        parts = [f"{name} {acts[name]:.1f}"
                 for name in ('Generation', 'Consumption', 'Distribution losses',
                               'Imports', 'Exports')
                 if name in acts]
        if parts:
            lines.append(f"{year}: " + ', '.join(parts))
    # Growth from earliest to latest year for headline activities
    years = sorted(by_year_activity)
    if len(years) >= 2:
        first_y, last_y = years[0], years[-1]
        deltas = []
        for activity in ('Generation', 'Consumption'):
            a0 = by_year_activity[first_y].get(activity)
            a1 = by_year_activity[last_y].get(activity)
            if a0 and a1 and a0 != 0:
                deltas.append(f"{activity} {(a1 - a0) / a0 * 100:+.1f}%")
        if deltas:
            lines.append(f"Growth {first_y}→{last_y}: " + ', '.join(deltas))
    return '\n'.join(lines)


_JSON_PARSERS: dict[str, Callable] = {
    'YahooFinanceCrude': _parse_yahoo,
    'YahooFinanceForex': _parse_yahoo,
    'OpenMeteoManila':   _parse_open_meteo,
    'WBPhilFood':        _parse_wb_phil_food,
    'EIAElectricity':    _parse_eia,
}


def _eia_api_key() -> Optional[str]:
    """Read EIA API key from env var, with fallback to ~/.ph_economic_ai/config.json."""
    key = os.environ.get('EIA_API_KEY')
    if key:
        return key.strip()
    cfg = Path.home() / '.ph_economic_ai' / 'config.json'
    if cfg.exists():
        try:
            data = json.loads(cfg.read_text(encoding='utf-8'))
            v = data.get('eia_api_key')
            if isinstance(v, str) and v.strip():
                return v.strip()
        except Exception as e:
            logging.warning('RagEngine: failed to read EIA key from %s: %s', cfg, e)
    return None


class RagEngine:
    SOURCES = SOURCES

    def __init__(self):
        self._chunks: list[Chunk] = []
        self._disabled: set[str] = set()
        # Semantic index (primary)
        self._embed_vecs: Optional[np.ndarray] = None   # (n_active, dim) float32, L2-normalised
        self._embed_active: list[Chunk] = []            # active chunks matching _embed_vecs rows
        self._embed_cache: dict[int, np.ndarray] = {}   # id(chunk) → normalised vector
        self._use_embeddings: bool = True               # flipped False if model unavailable
        # TF-IDF fallback (always built as safety net)
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._matrix = None  # scipy sparse
        self._dirty = False   # set when chunks added; cleared on _refit()

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch_all(
        self,
        on_progress: Optional[Callable[[str, str, int], None]] = None,
    ) -> dict[str, int]:
        """Fetch all sources in parallel. Returns {source_name: chunk_count}."""
        results: dict[str, int] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.SOURCES)) as pool:
            futures = {
                pool.submit(self._fetch_one, name, url): name
                for name, url in self.SOURCES.items()
            }
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                try:
                    chunks = future.result()
                    self._chunks.extend(chunks)
                    results[name] = len(chunks)
                except Exception as e:
                    logging.warning("RagEngine: fetch failed for %s: %s", name, e)
                    results[name] = 0
                if on_progress:
                    on_progress(name, 'done', results[name])
        self._refit()
        return results

    def add_pdf(self, path: str) -> int:
        """Load PDF with PyMuPDF, chunk it, add to index. Returns chunk count."""
        import fitz  # PyMuPDF
        try:
            doc = fitz.open(path)
        except Exception:
            raise
        try:
            text = '\n'.join(page.get_text() for page in doc)
        finally:
            doc.close()
        name = Path(path).stem
        chunks = self._chunk(text, source=name, url=path)
        self._chunks.extend(chunks)
        self._dirty = True  # refit deferred to next query()
        return len(chunks)

    def add_text(self, source: str, text: str, url: str = '') -> int:
        """Add pre-bundled corpus text. Returns chunk count."""
        new_chunks = self._chunk(text, source=source, url=url)
        self._chunks.extend(new_chunks)
        self._dirty = True  # refit deferred to next query()
        return len(new_chunks)

    def query(
        self,
        text: str,
        top_k: int = 5,
        sources: Optional[list[str]] = None,
    ) -> list[dict]:
        """Return top_k most relevant chunks. Uses semantic embeddings, falls back to TF-IDF."""
        if self._dirty:
            self._refit()
        if self._use_embeddings and self._embed_vecs is not None:
            try:
                return self._query_embeddings(text, top_k, sources)
            except Exception as e:
                logging.warning("RagEngine: embedding query failed (%s), using TF-IDF", e)
        return self._query_tfidf(text, top_k, sources)

    def _query_embeddings(self, text: str, top_k: int, sources: Optional[list[str]]) -> list[dict]:
        import ollama as _ol
        resp = _ol.embeddings(model=_EMBED_MODEL, prompt=text[:_EMBED_MAX_CHARS])
        q_vec = np.array(resp['embedding'], dtype=np.float32)
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec /= norm

        active = self._embed_active
        if sources:
            src_set = set(sources)
            idxs = [i for i, c in enumerate(active) if c.source in src_set]
            if not idxs:
                return []
            sub_vecs = self._embed_vecs[idxs]
            sub_chunks = [active[i] for i in idxs]
        else:
            sub_vecs = self._embed_vecs
            sub_chunks = active

        scores = sub_vecs @ q_vec   # cosine similarity (rows already L2-normalised)
        top_idxs = np.argsort(scores)[::-1][:top_k]
        return [
            {'source': sub_chunks[i].source, 'url': sub_chunks[i].url,
             'timestamp': sub_chunks[i].timestamp, 'text': sub_chunks[i].text,
             'score': float(scores[i])}
            for i in top_idxs if scores[i] > 0
        ]

    def _query_tfidf(self, text: str, top_k: int, sources: Optional[list[str]]) -> list[dict]:
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
            {'source': sub_chunks[i].source, 'url': sub_chunks[i].url,
             'timestamp': sub_chunks[i].timestamp, 'text': sub_chunks[i].text,
             'score': float(scores[i])}
            for i in top_idxs if scores[i] > 0
        ]

    def toggle_source(self, source: str, enabled: bool) -> None:
        if enabled:
            self._disabled.discard(source)
        else:
            self._disabled.add(source)
        self._dirty = True  # refit on next query(); avoids re-embedding all chunks

    @property
    def chunk_count(self) -> int:
        return len(self._active_chunks())

    @property
    def all_source_names(self) -> list[str]:
        return sorted({c.source for c in self._chunks})

    @property
    def source_chunk_counts(self) -> dict[str, int]:
        """Active chunk count per source (disabled sources excluded)."""
        from collections import Counter
        return dict(Counter(
            c.source for c in self._chunks if c.source not in self._disabled
        ))

    # ── Internal ──────────────────────────────────────────────────────────────

    def _fetch_one(self, name: str, url: str) -> list[Chunk]:
        if name in _JSON_SOURCES:
            return self._fetch_json(name, url)
        if name in _RSS_SOURCES:
            return self._fetch_rss(name, url)
        return self._fetch_html(name, url)

    def _fetch_html(self, name: str, url: str) -> list[Chunk]:
        resp = requests.get(url, headers=_HEADERS, timeout=12)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            tag.decompose()
        text = soup.get_text(separator=' ', strip=True)
        return self._chunk(text, source=name, url=url)

    def _fetch_rss(self, name: str, url: str) -> list[Chunk]:
        resp = requests.get(url, headers=_HEADERS, timeout=12)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        items = root.findall('.//item') or root.findall('.//atom:entry', ns)
        parts: list[str] = []
        for item in items[:20]:
            title = item.findtext('title') or item.findtext('atom:title', namespaces=ns) or ''
            desc = (item.findtext('description')
                    or item.findtext('atom:summary', namespaces=ns)
                    or item.findtext('atom:content', namespaces=ns)
                    or '')
            # Strip HTML tags from description
            if desc:
                desc = BeautifulSoup(desc, 'html.parser').get_text(separator=' ', strip=True)
            if title or desc:
                parts.append(f"{title}. {desc}".strip())
        text = '\n\n'.join(parts)
        return self._chunk(text, source=name, url=url) if text else []

    def _fetch_json(self, name: str, url: str) -> list[Chunk]:
        """Generic JSON fetch dispatching to per-source parser in _JSON_PARSERS."""
        # Per-source URL preprocessing
        if name == 'EIAElectricity':
            key = _eia_api_key()
            if not key:
                logging.info('RagEngine: EIA disabled (no EIA_API_KEY)')
                return []
            url = url.replace('{EIA_API_KEY}', key)
        resp = requests.get(url, headers=_JSON_HEADERS, timeout=15)
        resp.raise_for_status()
        parser = _JSON_PARSERS.get(name)
        if parser is None:
            return []
        text = parser(resp.json(), name)
        return self._chunk(text, source=name, url=url) if text else []

    def _chunk(self, text: str, source: str, url: str = '') -> list[Chunk]:
        step = max(1, _CHUNK_SIZE - _CHUNK_OVERLAP)
        timestamp = datetime.now(timezone.utc).isoformat()
        chunks = []
        start = 0
        while start < len(text):
            end = start + _CHUNK_SIZE
            chunks.append(Chunk(source=source, url=url, timestamp=timestamp, text=text[start:end]))
            start += step
        return chunks

    def _active_chunks(self) -> list[Chunk]:
        return [c for c in self._chunks if c.source not in self._disabled]

    def _refit(self) -> None:
        active = self._active_chunks()
        self._dirty = False
        if not active:
            self._vectorizer = None
            self._matrix = None
            self._embed_vecs = None
            self._embed_active = []
            return

        # ── TF-IDF (always built; used as fallback) ───────────────────────────
        self._vectorizer = TfidfVectorizer(max_features=10_000, stop_words='english')
        self._matrix = self._vectorizer.fit_transform([c.text for c in active])

        # ── Semantic embeddings (primary; cached so re-runs are instant) ───────
        if not self._use_embeddings:
            return
        try:
            new_chunks = [c for c in active if id(c) not in self._embed_cache]
            if new_chunks:
                self._embed_chunks_parallel(new_chunks)
            # Build matrix from cache in active order
            vecs = [self._embed_cache[id(c)] for c in active]
            self._embed_vecs = np.stack(vecs).astype(np.float32)
            self._embed_active = active
        except Exception as e:
            logging.warning("RagEngine: embedding unavailable (%s); using TF-IDF only", e)
            self._use_embeddings = False
            self._embed_vecs = None
            self._embed_active = []

    def _embed_chunks_parallel(self, chunks: list[Chunk]) -> None:
        """Embed chunks in parallel and store in cache. Raises on first failure."""
        import ollama as _ol

        def _embed_one(chunk: Chunk) -> tuple[int, np.ndarray]:
            resp = _ol.embeddings(model=_EMBED_MODEL, prompt=chunk.text[:_EMBED_MAX_CHARS])
            vec = np.array(resp['embedding'], dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            return id(chunk), vec

        with concurrent.futures.ThreadPoolExecutor(max_workers=_EMBED_WORKERS) as pool:
            futs = {pool.submit(_embed_one, c): c for c in chunks}
            for fut in concurrent.futures.as_completed(futs):
                chunk_id, vec = fut.result()   # raises on error → caught in _refit
                self._embed_cache[chunk_id] = vec
