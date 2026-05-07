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


if __name__ == "__main__":
    unittest.main()
