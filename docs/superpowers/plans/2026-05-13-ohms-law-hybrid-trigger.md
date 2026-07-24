# Ohm's Law Hybrid Widget Trigger — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the LLM write its own natural Ohm's Law response and signal the widget via an embedded tag, falling back to keyword detection if the tag is absent.

**Architecture:** Four changes to `Maria_App.py`: (1) add `_parse_ohms_tag` helper, (2) update the response handler to try the tag before the keyword fallback, (3) inject a tag instruction into the system prompt, (4) delete dead code. No new files, no schema changes, no UI changes.

**Tech Stack:** Python 3, `re`, `json` (both already imported in `Maria_App.py`)

---

## File Map

| File | Lines | Change |
|------|-------|--------|
| `Project_Maria/Maria_App.py` | ~38049 | Add `_parse_ohms_tag` method near other Ohm's Law helpers |
| `Project_Maria/Maria_App.py` | 46082–46088 | Replace keyword-only widget trigger with tag-first + keyword fallback |
| `Project_Maria/Maria_App.py` | 10556–10559 | Append tag instruction to base system prompt |
| `Project_Maria/Maria_App.py` | 38016–38100 | Delete `_is_ohms_law_widget_request`, `_build_ohms_law_widget_payload`, and their internal cross-call |

---

## Task 1: Add `_parse_ohms_tag`

**Files:**
- Modify: `Project_Maria/Maria_App.py` — add method after `_extract_ohms_law_values` (~line 38079)

- [ ] **Step 1: Verify the insertion point**

  Run:
  ```bash
  python -c "
  lines = open('Project_Maria/Maria_App.py', encoding='utf-8').readlines()
  for i, l in enumerate(lines[38049:38082], 38050):
      print(i, l, end='')
  "
  ```
  Confirm `_extract_ohms_law_values` ends around line 38079 and `_build_ohms_law_widget_payload` follows.

- [ ] **Step 2: Insert `_parse_ohms_tag` after `_extract_ohms_law_values`**

  Find the blank line immediately after `_extract_ohms_law_values` ends (after the `return {` block). Insert:

  ```python
      def _parse_ohms_tag(self, text: str):
          """Strip <!--ohms:{"v":...,"r":...}--> from LLM response.

          Returns (clean_text, widget_data) where widget_data is None if no
          valid tag was found.
          """
          m = re.search(r'<!--ohms:(\{[^}]+\})-->', text)
          if not m:
              return text, None
          try:
              data = json.loads(m.group(1))
              v = max(0.1, min(24.0, float(data.get('v', 9.0))))
              r = max(0.1, min(20.0, float(data.get('r', 3.0))))
              clean = (text[:m.start()].rstrip() + text[m.end():]).rstrip()
              return clean, {
                  'voltage':    round(v, 2),
                  'resistance': round(r, 1),
                  'current':    round(v / r, 4),
              }
          except Exception:
              return text, None
  ```

- [ ] **Step 3: Test `_parse_ohms_tag` in isolation**

  Run:
  ```bash
  python -c "
  import re, json

  def _parse_ohms_tag(text):
      m = re.search(r'<!--ohms:(\{[^}]+\})-->', text)
      if not m:
          return text, None
      try:
          data = json.loads(m.group(1))
          v = max(0.1, min(24.0, float(data.get('v', 9.0))))
          r = max(0.1, min(20.0, float(data.get('r', 3.0))))
          clean = (text[:m.start()].rstrip() + text[m.end():]).rstrip()
          return clean, {'voltage': round(v,2), 'resistance': round(r,1), 'current': round(v/r,4)}
      except Exception:
          return text, None

  # tag present — values extracted, tag stripped
  clean, data = _parse_ohms_tag('So the current is 3A.<!--ohms:{\"v\":9.0,\"r\":3.0}-->')
  assert clean == 'So the current is 3A.', repr(clean)
  assert data == {'voltage': 9.0, 'resistance': 3.0, 'current': 3.0}, data

  # no tag — text unchanged, None returned
  clean, data = _parse_ohms_tag('Just a normal response.')
  assert clean == 'Just a normal response.' and data is None

  # malformed JSON — graceful fallback
  clean, data = _parse_ohms_tag('Text <!--ohms:{broken json}-->')
  assert data is None

  # clamping — v=999 clamped to 24, r=0.001 clamped to 0.1
  clean, data = _parse_ohms_tag('x<!--ohms:{\"v\":999,\"r\":0.001}-->')
  assert data['voltage'] == 24.0 and data['resistance'] == 0.1

  print('All tests pass')
  "
  ```

  Expected output: `All tests pass`

- [ ] **Step 4: Smoke-check file parses**

  ```bash
  python -c "import ast; ast.parse(open('Project_Maria/Maria_App.py', encoding='utf-8').read()); print('OK')"
  ```

  Expected: `OK`

- [ ] **Step 5: Commit**

  ```bash
  git add Project_Maria/Maria_App.py
  git commit -m "feat: add _parse_ohms_tag to extract widget values from LLM response"
  ```

---

## Task 2: Wire tag-first into the response handler

**Files:**
- Modify: `Project_Maria/Maria_App.py:46082–46088`

- [ ] **Step 1: Find the exact block to replace**

  The current block at lines 46082–46088 reads:
  ```python
          if widget_payload is None and self._is_ohms_law_query(getattr(self, '_dpo_pending_prompt', '')):
              _auto_widget_data = self._extract_ohms_law_values(getattr(self, '_dpo_pending_prompt', ''))
              widget_payload = {
                  'has_widget': True,
                  'widget_type': 'ohms_law',
                  'widget_data': _auto_widget_data,
              }
  ```

- [ ] **Step 2: Replace it with the tag-first + fallback block**

  Replace the block above with:

  ```python
          # 1. LLM tag: strip <!--ohms:{...}--> and use those values
          response_text, _tag_data = self._parse_ohms_tag(response_text)
          if _tag_data:
              widget_payload = {
                  'has_widget': True,
                  'widget_type': 'ohms_law',
                  'widget_data': _tag_data,
              }
          # 2. Keyword fallback: prompt-based detection if LLM omitted the tag
          elif widget_payload is None and self._is_ohms_law_query(getattr(self, '_dpo_pending_prompt', '')):
              _auto_widget_data = self._extract_ohms_law_values(getattr(self, '_dpo_pending_prompt', ''))
              widget_payload = {
                  'has_widget': True,
                  'widget_type': 'ohms_law',
                  'widget_data': _auto_widget_data,
              }
  ```

- [ ] **Step 3: Smoke-check file parses**

  ```bash
  python -c "import ast; ast.parse(open('Project_Maria/Maria_App.py', encoding='utf-8').read()); print('OK')"
  ```

  Expected: `OK`

- [ ] **Step 4: Commit**

  ```bash
  git add Project_Maria/Maria_App.py
  git commit -m "feat: wire LLM tag as primary widget trigger with keyword fallback"
  ```

---

## Task 3: Inject tag instruction into system prompt

**Files:**
- Modify: `Project_Maria/Maria_App.py:10556–10559`

- [ ] **Step 1: Find the end of the base system prompt**

  The base system prompt string ends at the line containing only `"""` after the guidelines block. Confirm by reading around line 10559:
  ```bash
  python -c "
  lines = open('Project_Maria/Maria_App.py', encoding='utf-8').readlines()
  for i, l in enumerate(lines[10553:10563], 10554):
      print(i, repr(l))
  "
  ```
  You should see the closing `"""` of the system prompt triple-string.

- [ ] **Step 2: Add the tag instruction inside the system prompt string**

  Find this exact text inside the system prompt (the last guideline line):
  ```python
  - If asked to do something harmful or against your guidelines, decline briefly and offer to help with something else instead.
  """
  ```

  Replace it with:
  ```python
  - If asked to do something harmful or against your guidelines, decline briefly and offer to help with something else instead.

  Widget signal:
  - When you explain, calculate, or discuss Ohm's Law (V = IR / I = V/R), append this exact line at the very end of your response — after all your prose — substituting the voltage and resistance values you used or mentioned: <!--ohms:{"v":9.0,"r":3.0}-->
  - Use the values you actually discussed. Omit this line entirely if the topic is not about Ohm's Law.
  """
  ```

- [ ] **Step 3: Smoke-check file parses**

  ```bash
  python -c "import ast; ast.parse(open('Project_Maria/Maria_App.py', encoding='utf-8').read()); print('OK')"
  ```

  Expected: `OK`

- [ ] **Step 4: Commit**

  ```bash
  git add Project_Maria/Maria_App.py
  git commit -m "feat: instruct LLM to append ohms tag when discussing Ohm's Law"
  ```

---

## Task 4: Delete dead code

**Files:**
- Modify: `Project_Maria/Maria_App.py:38016–38100`

Delete two methods and one internal call that are no longer used.

- [ ] **Step 1: Delete `_is_ohms_law_widget_request`**

  Find and delete the entire method (lines ~38016–38030):
  ```python
      def _is_ohms_law_widget_request(self, user_text: str) -> bool:
          t = (user_text or '').lower()
          direct_terms = (
              "ohm's law", "ohms law", "ohm law",
              "voltage resistance current", "i = v / r", "i=v/r"
          )
          widget_terms = ("widget", "interactive", "slider", "simulation", "visualizer", "visualization")
          if any(term in t for term in direct_terms):
              return True
          return (
              ('ohm' in t or 'current' in t)
              and 'resistance' in t
              and 'voltage' in t
              and any(term in t for term in widget_terms)
          )
  ```

- [ ] **Step 2: Remove the call to `_is_ohms_law_widget_request` inside `_is_ohms_law_query`**

  Find this line inside `_is_ohms_law_query`:
  ```python
          if self._is_ohms_law_widget_request(user_text):
              return True
  ```
  Delete it (2 lines).

- [ ] **Step 3: Delete `_build_ohms_law_widget_payload`**

  Find and delete the entire method (lines ~38081–38100):
  ```python
      def _build_ohms_law_widget_payload(self, user_text: str) -> dict:
          widget_data = self._extract_ohms_law_values(user_text)
          explicit_widget = self._is_ohms_law_widget_request(user_text)
          lead = (
              f"I built an interactive Ohm's Law widget below. It starts at {widget_data['voltage']} V and "
              f"{widget_data['resistance']:.1f} Ω, so the current is {widget_data['current']:.2f} A. "
              "Move the sliders to see the result and the animated circuit update live."
              if explicit_widget else
              "Ohm's Law describes the relationship between voltage, current, and resistance in an electric circuit. "
              "The formula is `I = V / R`, which means current equals voltage divided by resistance. "
              "Higher voltage pushes more current through the circuit, while higher resistance restricts the flow.\n\n"
              f"I also opened the live widget below using {widget_data['voltage']} V and {widget_data['resistance']:.1f} Ω "
              f"as the working values. That gives {widget_data['current']:.2f} A, and you can adjust the model in real time."
          )
          return {
              'text': lead,
              'has_widget': True,
              'widget_type': 'ohms_law',
              'widget_data': widget_data,
          }
  ```

- [ ] **Step 4: Verify no remaining references**

  ```bash
  python -c "
  text = open('Project_Maria/Maria_App.py', encoding='utf-8').read()
  for name in ['_is_ohms_law_widget_request', '_build_ohms_law_widget_payload']:
      count = text.count(name)
      print(f'{name}: {count} occurrences (expected 0)')
  "
  ```

  Expected:
  ```
  _is_ohms_law_widget_request: 0 occurrences (expected 0)
  _build_ohms_law_widget_payload: 0 occurrences (expected 0)
  ```

- [ ] **Step 5: Smoke-check file parses**

  ```bash
  python -c "import ast; ast.parse(open('Project_Maria/Maria_App.py', encoding='utf-8').read()); print('OK')"
  ```

  Expected: `OK`

- [ ] **Step 6: Commit**

  ```bash
  git add Project_Maria/Maria_App.py
  git commit -m "refactor: remove dead _is_ohms_law_widget_request and _build_ohms_law_widget_payload"
  ```
