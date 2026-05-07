import json
import os
import sys
import unittest

KB_PATH = os.path.join(os.path.dirname(__file__), "OfflineDisasterKB.json")


class TestOfflineDisasterKB(unittest.TestCase):

    def setUp(self):
        with open(KB_PATH, encoding="utf-8") as f:
            self.kb = json.load(f)

    def test_typhoon_has_all_signals(self):
        typhoon = self.kb["typhoon"]
        for i in range(1, 6):
            key = f"signal_{i}"
            self.assertIn(key, typhoon, f"Missing {key}")
            self.assertIn("meaning", typhoon[key])
            self.assertIn("what_to_do", typhoon[key])

    def test_earthquake_has_three_keys(self):
        eq = self.kb["earthquake"]
        for key in ("during", "after", "tsunami_watch"):
            self.assertIn(key, eq)
            self.assertTrue(len(eq[key]) > 10, f"{key} content too short")

    def test_volcanic_has_all_levels(self):
        volcanic = self.kb["volcanic"]
        for i in range(1, 6):
            key = f"alert_{i}"
            self.assertIn(key, volcanic, f"Missing {key}")

    def test_go_bag_is_list(self):
        self.assertIsInstance(self.kb["go_bag"], list)
        self.assertGreaterEqual(len(self.kb["go_bag"]), 5)

    def test_hotlines_have_name_and_number(self):
        for h in self.kb["hotlines"]:
            self.assertIn("name", h)
            self.assertIn("number", h)
            self.assertTrue(len(h["number"]) >= 3)


import re as _re

_SIGNAL_RE = _re.compile(
    r'(?:Public\s+Storm\s+Warning\s+)?(?:Signal\s+(?:No\.?\s*)?#?\s*|PSWS\s+#\s*)([1-5])',
    _re.IGNORECASE
)
_TYPHOON_RE = _re.compile(r'[Tt]yphoon\s+([A-Z][a-z]+)|[Bb]agyo\s+([A-Z][a-z]+)')
_EQ_MIN_MAG = 4.0


def _PHDisasterWatcher_parse_pagasa_html(html: str):
    m = _SIGNAL_RE.search(html)
    if not m:
        return None
    signal = int(m.group(1))
    name_m = _TYPHOON_RE.search(html)
    typhoon_name = (name_m.group(1) or name_m.group(2)) if name_m else ""
    return {"type": "typhoon", "signal": signal, "typhoon_name": typhoon_name}


def _PHDisasterWatcher_parse_phivolcs_json(data: list):
    if not data:
        return None
    latest = data[0]
    try:
        mag = float(latest.get("Magnitude", 0))
    except (ValueError, TypeError):
        return None
    if mag < _EQ_MIN_MAG:
        return None
    return {
        "type": "earthquake",
        "magnitude": mag,
        "depth": latest.get("Depth", ""),
        "location": latest.get("Location", ""),
        "date": latest.get("Date", ""),
        "tsunami_watch": mag >= 7.0,
    }


class TestPHDisasterWatcherParsing(unittest.TestCase):

    def test_parse_pagasa_signal_2(self):
        html = "<p>Public Storm Warning Signal No. 2 is raised over Metro Manila</p>"
        result = _PHDisasterWatcher_parse_pagasa_html(html)
        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "typhoon")
        self.assertEqual(result["signal"], 2)

    def test_parse_pagasa_signal_with_hash(self):
        html = "PSWS #3 is in effect for Rizal, Quezon, Aurora"
        result = _PHDisasterWatcher_parse_pagasa_html(html)
        self.assertIsNotNone(result)
        self.assertEqual(result["signal"], 3)

    def test_parse_pagasa_no_signal_returns_none(self):
        html = "<p>No tropical cyclone affecting any part of the Philippines.</p>"
        result = _PHDisasterWatcher_parse_pagasa_html(html)
        self.assertIsNone(result)

    def test_parse_pagasa_extracts_typhoon_name(self):
        html = "Signal No. 2 — Typhoon Carina is 300 km east of Aurora"
        result = _PHDisasterWatcher_parse_pagasa_html(html)
        self.assertEqual(result["typhoon_name"], "Carina")

    def test_parse_phivolcs_m5_earthquake(self):
        data = [{"Magnitude": "5.1", "Depth": "15 km",
                 "Location": "10 km N of Davao City", "Date": "07 May 2026 - 02:15 AM"}]
        result = _PHDisasterWatcher_parse_phivolcs_json(data)
        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "earthquake")
        self.assertAlmostEqual(result["magnitude"], 5.1)
        self.assertFalse(result["tsunami_watch"])

    def test_parse_phivolcs_m7_sets_tsunami_watch(self):
        data = [{"Magnitude": "7.2", "Depth": "10 km",
                 "Location": "50 km W of Davao", "Date": "07 May 2026"}]
        result = _PHDisasterWatcher_parse_phivolcs_json(data)
        self.assertTrue(result["tsunami_watch"])

    def test_parse_phivolcs_below_threshold_returns_none(self):
        data = [{"Magnitude": "2.5", "Depth": "5 km",
                 "Location": "Somewhere", "Date": "07 May 2026"}]
        result = _PHDisasterWatcher_parse_phivolcs_json(data)
        self.assertIsNone(result)

    def test_parse_phivolcs_empty_list_returns_none(self):
        self.assertIsNone(_PHDisasterWatcher_parse_phivolcs_json([]))


if __name__ == "__main__":
    unittest.main()
