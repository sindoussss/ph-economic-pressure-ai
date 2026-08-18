"""Build `benchmark/data/doe_price_adjustments.csv` from DOE's weekly notices.

The announced pump move, which is the top of the accuracy roadmap: for six days
of every seven, "what will fuel cost next week" is a published fact and the app
was guessing at it.

**Why this needs a browser when nothing else in the repo does.** The notice is a
PDF, but the link to that PDF is written into the page by JavaScript after
hydration. It is absent from the served HTML, from the Nuxt payload and from the
static document list, so `requests` alone cannot find it. The Liferay content API
holds the article (`structured-contents/3617975` answers with a permission error
naming it, so the ID is right) but serves nothing anonymously, and the custom
`api-prod.doe.gov.ph` API exposes no article endpoint. Rendering the page is the
only route left that does not involve credentials.

Playwright is imported lazily and only here. Nothing else in the package touches
it, CI never installs it, and the tests read a committed fixture -- the same
fetch-once-commit contract every other panel in `benchmark/data` keeps.

    pip install playwright && playwright install chromium
    python -m ph_economic_ai.tools.refresh_doe_adjustment
"""
from __future__ import annotations

import csv
import datetime as dt
import re
from typing import Optional

import requests

from ph_economic_ai.benchmark.doe_adjustment import (
    industry_adjustment, parse_notice_pdf,
)
from ph_economic_ai.benchmark.paths import data
from ph_economic_ai.benchmark.provenance import write_record

OUT = data('doe_price_adjustments.csv')

LISTING = ('https://doe.gov.ph/site/oimb/articles/group/liquid-fuels'
           '?display_type=Card')
BASE = 'https://doe.gov.ph'
HEADERS = {'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/120.0.0.0 Safari/537.36')}

#: The consent gate blocks rendering until dismissed. This is the narrowest of
#: the three offered options -- essential and session cookies only, one day --
#: chosen over either "Accept All Cookies" variant.
CONSENT_ESSENTIAL_ONLY = 'Allow only Essential and Session Cookies'

_NOTICE_HREF = re.compile(r'href=["\'](/articles/\d+--prior-notice[^"\']*)')
_DOC_HREF = re.compile(r'https://prod-cms\.doe\.gov\.ph/documents/[^"\'\s]+')


def discover_notices() -> list[str]:
    """Article URLs for the published Prior Notice pages, newest listing first.

    Plain `requests`: the LISTING is server-rendered even though the article
    bodies are not.
    """
    resp = requests.get(LISTING, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    seen, out = set(), []
    for href in _NOTICE_HREF.findall(resp.text):
        path = href.split('?')[0]
        if path not in seen:
            seen.add(path)
            out.append(BASE + path)
    return out


def pdf_url_for(article_url: str, timeout_ms: int = 90000) -> Optional[str]:
    """The notice PDF linked from a rendered article page, or None."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as play:
        browser = play.chromium.launch()
        try:
            page = browser.new_page(user_agent=HEADERS['User-Agent'])
            page.goto(article_url, wait_until='domcontentloaded', timeout=timeout_ms)
            try:
                page.get_by_text(CONSENT_ESSENTIAL_ONLY, exact=False).first.click(
                    timeout=8000)
            except Exception:
                pass                      # gate absent once the choice is stored
            page.wait_for_load_state('networkidle', timeout=timeout_ms)
            hrefs = page.eval_on_selector_all('a', 'els => els.map(e => e.href)')
        finally:
            browser.close()

    for href in hrefs or []:
        if href and 'prod-cms.doe.gov.ph/documents/' in href and 'itmsfuel' in href.lower():
            return href
    for href in hrefs or []:
        if href and _DOC_HREF.fullmatch(href) and 'pdf' in href.lower():
            return href
    return None


def fetch_notice(article_url: str) -> Optional[dict]:
    """One notice, from article URL to an industry adjustment."""
    pdf_url = pdf_url_for(article_url)
    if not pdf_url:
        print(f'  no PDF link found on {article_url[-60:]}')
        return None
    blob = requests.get(pdf_url, headers=HEADERS, timeout=90)
    blob.raise_for_status()
    parsed = parse_notice_pdf(blob.content)
    if not parsed['week']:
        print(f'  no readable week heading in {pdf_url[-50:]}')
        return None
    result = industry_adjustment(parsed['rows'], summary=parsed['summary'])
    start, end = parsed['week']
    return {'week_start': start.isoformat(), 'week_end': end.isoformat(),
            'gasoline': result['gasoline'], 'diesel': result['diesel'],
            'kerosene': result['kerosene'], 'basis': result['basis'],
            'n_companies': result['n_companies'],
            'consensus': result['consensus'], 'source_pdf': pdf_url}


def build() -> None:
    articles = discover_notices()
    print(f'{len(articles)} notice article(s) listed')
    rows = []
    for url in articles:
        print(f'- {url[-64:]}')
        try:
            row = fetch_notice(url)
        except Exception as exc:
            print(f'  FAILED {type(exc).__name__}: {exc}')
            continue
        if row:
            rows.append(row)
            print(f"  {row['week_start']}  gasoline {row['gasoline']}  "
                  f"diesel {row['diesel']}  ({row['basis']})")

    if not rows:
        raise SystemExit('no notices parsed; nothing written')

    existing: dict[str, dict] = {}
    if OUT.exists():
        with open(OUT, encoding='utf-8') as fh:
            existing = {r['week_start']: r for r in csv.DictReader(fh)}
    for row in rows:
        existing[row['week_start']] = {k: ('' if v is None else v)
                                       for k, v in row.items()}

    fields = ['week_start', 'week_end', 'gasoline', 'diesel', 'kerosene',
              'basis', 'n_companies', 'consensus', 'source_pdf']
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', encoding='utf-8', newline='\n') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for key in sorted(existing):
            writer.writerow({f: existing[key].get(f, '') for f in fields})

    write_record(
        OUT,
        source=('DOE Oil Industry Management Bureau, Prior Notice on Price '
                'Adjustments (weekly PDF linked from a client-rendered article)'),
        params={'listing': LISTING,
                'render': 'playwright chromium, maintainer machine only',
                'consent': CONSENT_ESSENTIAL_ONLY},
        transformations=[
            'discover notice articles from the server-rendered listing',
            'render each article to recover the PDF link written by JavaScript',
            'extract text with pypdf; read the week from the heading',
            'industry figure from the summary row where present, else the modal '
            'company figure; mode not mean so one company cannot drag it',
        ],
        units='PHP per litre, change for the week',
        notes=('The announced move, not a forecast. Announced Monday, effective '
               '6:00 AM Tuesday, and holds for the week.'),
    )
    print(f'\nWrote {OUT.name} ({len(existing)} week(s)) + provenance')


if __name__ == '__main__':
    build()
