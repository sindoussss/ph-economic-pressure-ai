"""Grounded entity/relation extraction over REAL chunk text. Pure parser is the
tested seam; the LLM call degrades to an empty result on any failure so the
structured graph is never broken by this optional layer."""
import logging
import re

from ph_economic_ai.engine import llm

_TIER = llm.FAST                        # extraction is mechanical — no deep tier needed
_PROMPT = (
    'Extract named entities and relations from the Philippine economic text. '
    'Output ONLY lines in these exact formats, nothing else:\n'
    'ENTITY: <name> | <type>   (type one of: commodity, agency, place, policy, figure)\n'
    'REL: <a> -> <b> | <relation>\n'
)

_ENTITY_RE = re.compile(r'^\s*ENTITY:\s*(.+?)\s*\|\s*(.+?)\s*$')
_REL_RE = re.compile(r'^\s*REL:\s*(.+?)\s*->\s*(.+?)\s*\|\s*(.+?)\s*$')


def parse_extraction(text: str) -> dict:
    entities, relations = [], []
    for line in (text or '').splitlines():
        m = _ENTITY_RE.match(line)
        if m and m.group(1).strip():
            entities.append({'name': m.group(1).strip(), 'type': m.group(2).strip()})
            continue
        m = _REL_RE.match(line)
        if m and m.group(1).strip() and m.group(2).strip():
            relations.append({'a': m.group(1).strip(), 'b': m.group(2).strip(),
                              'kind': m.group(3).strip()})
    return {'entities': entities, 'relations': relations}


def extract(chunk_text: str, source: str = '', tier: str = _TIER) -> dict:
    if not (chunk_text or '').strip() or not llm.is_configured():
        return {'entities': [], 'relations': []}
    try:
        text = llm.complete(
            [{'role': 'system', 'content': _PROMPT},
             {'role': 'user', 'content': chunk_text[:1500]}],
            tier=tier,
            max_tokens=256,
        )
        return parse_extraction(text)
    except Exception as exc:                      # noqa: BLE001 — must never raise
        logging.warning('entity extraction failed for %s: %s', source, exc)
        return {'entities': [], 'relations': []}
