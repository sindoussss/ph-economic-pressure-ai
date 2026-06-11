"""Grounded entity/relation extraction over REAL chunk text. Pure parser is the
tested seam; the ollama call degrades to an empty result on any failure so the
structured graph is never broken by this optional layer."""
import logging
import re

try:
    import ollama
except Exception:                       # ollama not installed / importable
    ollama = None

_MODEL = 'qwen2.5:3b'                    # already pulled for the swarm — no new dep
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


def extract(chunk_text: str, source: str = '', model: str = _MODEL) -> dict:
    if ollama is None or not (chunk_text or '').strip():
        return {'entities': [], 'relations': []}
    try:
        resp = ollama.chat(
            model=model,
            messages=[{'role': 'system', 'content': _PROMPT},
                      {'role': 'user', 'content': chunk_text[:1500]}],
            options={'num_predict': 256, 'temperature': 0.1},
        )
        return parse_extraction(resp['message']['content'])
    except Exception as exc:                      # noqa: BLE001 — must never raise
        logging.warning('entity extraction failed for %s: %s', source, exc)
        return {'entities': [], 'relations': []}
