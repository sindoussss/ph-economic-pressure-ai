# Ohm's Law Hybrid Widget Trigger

**Date:** 2026-05-13  
**Status:** Approved

## Problem

The Ohm's Law widget is triggered entirely by keyword/regex matching on the user's prompt. The LLM plays no role — it never knows the widget will appear, can't choose the values, and can't opt out. The post-LLM keyword check also hardcodes the canned response text in `_build_ohms_law_widget_payload` (dead code, never called) and provides no way for the LLM's own explanation to drive the widget values.

## Goal

Let the LLM write its response naturally AND signal the widget via an embedded tag. If the LLM omits the tag, fall back to the existing keyword detection. The LLM is the first choice; regex is the safety net.

## Approach: Prompt Tag + Fallback

### 1. System prompt instruction

Append the following paragraph to Maria's base system prompt (after the existing guidelines block, before any per-session injections):

```
When you explain, calculate, or discuss Ohm's Law (V = IR / I = V/R), append this exact line at the very end of your response — after all your prose — substituting the voltage and resistance values you used or mentioned:

<!--ohms:{"v":9.0,"r":3.0}-->

Use the values you actually discussed. Omit this line entirely if the topic is not about Ohm's Law.
```

### 2. New method: `_parse_ohms_tag(text: str) -> tuple[str, dict | None]`

Scans the response text for `<!--ohms:{...}-->`. Returns `(clean_text, widget_data)` where:
- `clean_text` has the tag stripped (and any trailing whitespace the tag left behind)
- `widget_data` is `{"voltage": v, "resistance": r, "current": round(v/r, 4)}` with V and R clamped to `[0.1, 24.0]` and `[0.1, 20.0]` respectively
- If no tag is found or the JSON is malformed, returns `(text, None)`

Regex: `r'<!--ohms:(\{[^}]+\})-->'`

### 3. Response handler update (~line 46075)

Current flow:
```python
widget_payload = response if isinstance(response, dict) and response.get('has_widget') else None
response_text = ...
if widget_payload is None and self._is_ohms_law_query(pending_prompt):
    widget_payload = { 'has_widget': True, 'widget_type': 'ohms_law', ... }
```

New flow:
```python
widget_payload = response if isinstance(response, dict) and response.get('has_widget') else None
response_text = ...

# 1. Try LLM tag first
response_text, _tag_data = self._parse_ohms_tag(response_text)
if _tag_data:
    widget_payload = {'has_widget': True, 'widget_type': 'ohms_law', 'widget_data': _tag_data}

# 2. Fallback: keyword detection on the original prompt
elif widget_payload is None and self._is_ohms_law_query(getattr(self, '_dpo_pending_prompt', '')):
    _auto_data = self._extract_ohms_law_values(getattr(self, '_dpo_pending_prompt', ''))
    widget_payload = {'has_widget': True, 'widget_type': 'ohms_law', 'widget_data': _auto_data}
```

### 4. Dead code removal

Delete the following (they are defined but never called):
- `_build_ohms_law_widget_payload` (~line 38081)
- `_is_ohms_law_widget_request` (~line 38016) — also called only from `_is_ohms_law_query` and `_build_ohms_law_widget_payload`; removing the call inside `_is_ohms_law_query` makes it fully unused

`_is_ohms_law_query` and `_extract_ohms_law_values` are **kept** — they are the fallback.

## What Does NOT Change

- `_is_ohms_law_query` — kept, is the fallback layer
- `_extract_ohms_law_values` — kept, used by the fallback
- `_insert_ohms_law_chat_widget`, `_OhmsLawWidget`, `_OhmsLawOscilloscope` — untouched
- `widget_data` / `apply_widget_data` / `state_changed` — untouched

## Files Affected

| File | Change |
|------|--------|
| `Project_Maria/Maria_App.py:10537–10559` | Append tag instruction to system prompt |
| `Project_Maria/Maria_App.py:38016–38100` | Delete `_is_ohms_law_widget_request` and `_build_ohms_law_widget_payload`; remove the call to `_is_ohms_law_widget_request` inside `_is_ohms_law_query` |
| `Project_Maria/Maria_App.py:~46075` | Add `_parse_ohms_tag` call before fallback keyword check |
| New method `_parse_ohms_tag` | Add near the other Ohm's Law helpers |
