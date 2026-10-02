import tempfile
import unittest
from pathlib import Path

import numpy as np

from ravel.io import load_physics_json, save_physics_json
from ravel.optimize import evaluate_dictionary
from ravel.presets import load_initial_atom_presets
from ravel.synthetic import make_dynamic_route_cohort


class OptimizerAndIOTest(unittest.TestCase):
    def test_truth_dictionary_beats_nuisance(self):
        sessions, atoms = make_dynamic_route_cohort(3)
        objective, table = evaluate_dictionary(
            sessions, atoms, "active", "val_select"
        )
        self.assertTrue(np.isfinite(objective))
        self.assertLess(objective, 0.0)
        self.assertTrue((table["delta_nll_vs_nuisance"] < 0.0).all())

    def test_physics_json_round_trip(self):
        _, atoms = make_dynamic_route_cohort(1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "physics.json"
            save_physics_json("X_to_Y", atoms, path, metadata={"frozen": True})
            route, loaded, metadata = load_physics_json(path)
        self.assertEqual(route, "X_to_Y")
        self.assertEqual(loaded, atoms)
        self.assertTrue(metadata["frozen"])

    def test_development_preset_loader(self):
        payload = {
            "SYNTHETIC_A_to_B": {
                "2": [
                    {"kind": "pole1", "delay_ms": 8.0, "pole_hz": 20.0},
                    {"kind": "pole1", "delay_ms": 16.0, "pole_hz": 12.0},
                ]
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preset.json"
            import json
            path.write_text(json.dumps(payload))
            presets = load_initial_atom_presets(path, "SYNTHETIC_A_to_B")
        self.assertEqual(len(presets[2]), 2)
        self.assertAlmostEqual(presets[2][1].delay_ms, 16.0)


if __name__ == "__main__":
    unittest.main()
