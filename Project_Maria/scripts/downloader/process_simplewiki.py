"""
Process Simple English Wikipedia XML dump → SQLite database.

Run this ONCE after downloading the dump with wikipedia.py.

Usage:
    python process_simplewiki.py

Reads:  ../../simplewiki-latest-pages-articles.xml.bz2
Writes: ../../simplewiki_articles.db

Requires:
    pip install mwparserfromhell tqdm
    (mwparserfromhell gives cleaner text; falls back to regex if not installed)
"""
import bz2
import os
import re
import sqlite3
import xml.etree.ElementTree as ET
from tqdm import tqdm

try:
    import mwparserfromhell
    HAS_MWP = True
except ImportError:
    HAS_MWP = False
    print("⚠️  mwparserfromhell not found — using regex fallback (less clean)")
    print("   Install with:  pip install mwparserfromhell\n")

BASE     = os.path.dirname(os.path.abspath(__file__))
DUMP_FILE = os.path.join(BASE, "..", "simplewiki-latest-pages-articles.xml.bz2")
DB_FILE   = os.path.join(BASE, "..", "..", "simplewiki_articles.db")
def strip_markup(wikitext: str) -> str:
    if HAS_MWP:
        try:
            return mwparserfromhell.parse(wikitext).strip_code()
        except Exception:
            pass
    # Regex fallback
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', wikitext)
    text = re.sub(r'\{\{[^}]*\}\}', '', text)
    text = re.sub(r'==+[^=]+=+=', '', text)
    text = re.sub(r"'{2,}", '', text)
    text = re.sub(r'<[^>]+>', '', text)
    return text.strip()


def process_dump(dump_file: str, db_file: str):
    dump_file = os.path.abspath(dump_file)
    db_file   = os.path.abspath(db_file)

    print(f"Reading:  {dump_file}")
    print(f"Writing:  {db_file}\n")

    if not os.path.exists(dump_file):
        print(f"ERROR: Dump file not found.\nExpected: {dump_file}")
        print("Run wikipedia.py first to download it.")
        return

    con = sqlite3.connect(db_file)
    con.execute("CREATE TABLE IF NOT EXISTS articles (title TEXT, text TEXT)")
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_articles_title_lower "
        "ON articles (LOWER(title))"
    )
    con.commit()

    batch: list = []
    count = 0
    ns_prefix = ''   # detected from first element

    with bz2.open(dump_file, 'rb') as f:
        for _event, elem in tqdm(ET.iterparse(f, events=('end',)), desc="Pages", unit=" pg"):
            # Auto-detect namespace from the first tag we see
            if not ns_prefix and elem.tag.startswith('{'):
                ns_prefix = elem.tag.split('}')[0] + '}'

            tag = elem.tag.replace(ns_prefix, '') if ns_prefix else elem.tag

            if tag == 'page':
                ns_elem    = elem.find(f'{ns_prefix}ns')
                title_elem = elem.find(f'{ns_prefix}title')
                text_elem  = elem.find(f'.//{ns_prefix}text')

                # Main namespace only (ns=0)
                if (ns_elem is not None and ns_elem.text == '0'
                        and title_elem is not None
                        and text_elem is not None):

                    wikitext = text_elem.text or ''

                    # Skip redirects
                    if wikitext.strip().lower().startswith('#redirect'):
                        elem.clear()
                        continue

                    clean = strip_markup(wikitext)
                    if len(clean) > 100:
                        batch.append((title_elem.text or '', clean))
                        count += 1

                        if len(batch) >= 1000:
                            con.executemany("INSERT INTO articles VALUES (?, ?)", batch)
                            con.commit()
                            batch.clear()

                elem.clear()

    if batch:
        con.executemany("INSERT INTO articles VALUES (?, ?)", batch)
        con.commit()

    con.close()
    print(f"\n✅ Done! {count:,} articles written to:\n   {db_file}")


if __name__ == "__main__":
    process_dump(DUMP_FILE, DB_FILE)
