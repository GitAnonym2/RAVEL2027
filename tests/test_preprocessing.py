import unittest

import numpy as np
import pandas as pd

from ravel.preprocess import (
    build_route_session_from_epochs,
    line_noise_ratio,
    zero_phase_notch,
)


class PreprocessingTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        trials, samples, channels = 40, 1500, 4
        t = np.arange(samples) / 500.0
        source = rng.normal(scale=0.5, size=(trials, samples, channels))
        source += 2.0 * np.sin(2 * np.pi * 60.0 * t)[None, :, None]
        target = np.roll(source, 4, axis=1) + rng.normal(
            scale=0.2, size=source.shape
        )
        self.source = source.astype(np.float32)
        self.target = target.astype(np.float32)
        self.metadata = pd.DataFrame(
            {"outcome": np.where(np.arange(trials) % 2, "hit", "miss")}
        )

    def test_notch_reduces_line_ratio(self):
        before = line_noise_ratio(self.source)
        after = line_noise_ratio(zero_phase_notch(self.source))
        self.assertLess(after, 0.05 * before)

    def test_epoch_builder_obeys_frequency_contract(self):
        session = build_route_session_from_epochs(
            source_epochs={"active": self.source},
            target_epochs={"active": self.target},
            metadata={"active": self.metadata},
            route_name="X_to_Y",
            source_region="X",
            target_region="Y",
            session_id=9,
            subject_id="mouse9",
            d=2,
        )
        self.assertGreaterEqual(session.freqs_hz.min(), 6.0)
        self.assertLessEqual(session.freqs_hz.max(), 80.0)
        self.assertFalse(
            np.any((session.freqs_hz >= 58.0) & (session.freqs_hz <= 62.0))
        )
        self.assertEqual(session.contexts["active"].source_coord.shape[-1], 2)
        split_union = np.concatenate(
            list(session.contexts["active"].splits.values())
        )
        self.assertEqual(len(np.unique(split_union)), len(self.metadata))


if __name__ == "__main__":
    unittest.main()

