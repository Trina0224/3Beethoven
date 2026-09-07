import copy
import json
import math
import unittest
import tempfile
from pathlib import Path

from prepare_stats_consolidation_teacher import gate_acceptance
from run_stats_consolidation_compare import gate_result, save_immutable_contract


def metric(categories, correct=None, pending=0, depth3=None):
    correct = correct or categories
    depth3 = depth3 or {key: 0 for key in categories}
    return {"n": sum(categories.values()), "correct": sum(correct.values()), "pending": pending,
            "category_counts": {key: {"n": n, "correct": correct[key],
                                       "depth3_n": depth3.get(key, 0),
                                       "depth3_correct": depth3.get(key, 0)}
                                for key, n in categories.items()}}


class ProtocolGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open("docs/STATS_CONSOLIDATION_DECISION_DRAFT.json") as handle:
            cls.config = json.load(handle)

    def test_pending_never_passes_selection(self):
        old = {f"o{i}": 6 for i in range(8)}
        chain = {f"c{i}": 12 for i in range(4)}
        event = {name: 4 for name in ("exactly_one", "both", "neither", "same")}
        baseline = {"old": metric(old), "chain": metric(chain, depth3={k: 4 for k in chain}),
                    "event": metric(event)}
        candidate = copy.deepcopy(baseline)
        candidate["old"]["pending"] = 1
        self.assertFalse(gate_result(candidate, baseline, self.config["validation_gate"])["passed"])

    def test_relative_losses_and_event_integer_floors(self):
        old = {f"o{i}": 6 for i in range(8)}
        chain = {f"c{i}": 12 for i in range(4)}
        event = {name: 4 for name in ("exactly_one", "both", "neither", "same")}
        baseline = {"old": metric(old), "chain": metric(chain, depth3={k: 4 for k in chain}), "event": metric(event)}
        self.assertTrue(gate_result(copy.deepcopy(baseline), baseline, self.config["validation_gate"])["passed"])
        failed = copy.deepcopy(baseline)
        failed["event"]["category_counts"]["same"]["correct"] = 2
        failed["event"]["correct"] = 14
        self.assertFalse(gate_result(failed, baseline, self.config["validation_gate"])["passed"])

    def test_pilot_archetype_floor_and_full_category_floor(self):
        teacher = self.config["teacher"]
        pilot = {"scope": "pilot", "accepted": 59, "pending_ids": ["x"] * 6,
                 "unresolved_ids": [],
                 "planned_by_archetype": {"a": 5, "b": 10},
                 "accepted_by_archetype": {"a": 4, "b": 8},
                 "planned_by_category": {}, "accepted_by_category": {}}
        self.assertTrue(gate_acceptance(pilot, teacher)["passed"])
        pilot["unresolved_ids"] = ["needs_semantic_review"]
        self.assertFalse(gate_acceptance(pilot, teacher)["passed"])
        pilot["unresolved_ids"] = []
        pilot["accepted_by_archetype"]["a"] = 3
        self.assertFalse(gate_acceptance(pilot, teacher)["passed"])
        full = {"scope": "full", "accepted": 0, "pending_ids": [],
                "planned_by_archetype": {}, "accepted_by_archetype": {},
                "planned_by_category": {"x": 6}, "accepted_by_category": {"x": 4}}
        self.assertFalse(gate_acceptance(full, teacher)["passed"])

    def test_contract_resume_accepts_exact_match_and_rejects_drift(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "contract.json"
            save_immutable_contract(path, {"seed": 2027, "parent": None})
            save_immutable_contract(path, {"seed": 2027, "parent": None})
            with self.assertRaises(RuntimeError):
                save_immutable_contract(path, {"seed": 31415, "parent": None})


if __name__ == "__main__":
    unittest.main()
