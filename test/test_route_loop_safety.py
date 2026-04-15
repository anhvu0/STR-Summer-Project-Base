import unittest
from collections import deque

from core.route_loop_safety import (
    dead_end_reentry_count,
    has_short_cycle_repeat,
    is_aba_bounce,
    transition_signal,
)


class RouteLoopSafetyTests(unittest.TestCase):
    def test_detects_aba_bounce(self):
        self.assertTrue(is_aba_bounce(["A", "B", "A"]))
        self.assertFalse(is_aba_bounce(["A", "B", "C"]))

    def test_detects_short_cycle_repeat(self):
        self.assertTrue(has_short_cycle_repeat(["A", "B", "C", "A", "B", "C"], max_cycle_len=3))
        self.assertFalse(has_short_cycle_repeat(["A", "B", "C", "D"]))

    def test_dead_end_reentry_count(self):
        out_deg = {"A": 0, "B": 3}
        self.assertEqual(dead_end_reentry_count(["A", "A", "B", "A"], out_deg), 2)
        self.assertEqual(dead_end_reentry_count(["B", "B", "A"], {"B": 1, "A": 2}), 0)

    def test_transition_signal(self):
        history = deque(["X", "Y"], maxlen=8)
        signals = transition_signal(history, "X", edge_out_degree={"X": 1, "Y": 2})
        self.assertTrue(signals["aba_bounce"])
        self.assertTrue(signals["forced_corridor"])


if __name__ == "__main__":
    unittest.main()
