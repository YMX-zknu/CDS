import importlib.util
from pathlib import Path
import sys
import unittest


HAS_TORCH = importlib.util.find_spec("torch") is not None
ROOT = Path(__file__).resolve().parents[1]
BOUNDARY_DIR = ROOT / "experiments" / "mechanism_boundary"
sys.path.insert(0, str(BOUNDARY_DIR))


@unittest.skipUnless(HAS_TORCH, "PyTorch is required for the mechanism-boundary checks")
class MechanismBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import context_relevance_boundary as boundary

        cls.boundary = boundary
        cls.spec = boundary.BoundarySpec()

    def test_samplewise_event_count_and_amplitude_matching(self):
        import numpy as np

        for regime in self.boundary.REGIMES:
            with self.subTest(regime=regime):
                batch = self.boundary.generate_boundary_batch(32, regime, 17, self.spec)
                relevant_count = batch.relevant_mask.sum(axis=(1, 2))
                distractor_count = batch.distractor_mask.sum(axis=(1, 2))
                np.testing.assert_array_equal(relevant_count, distractor_count)
                relevant_amplitude = (batch.inputs * batch.relevant_mask).sum(axis=(1, 2))
                distractor_amplitude = (batch.inputs * batch.distractor_mask).sum(axis=(1, 2))
                np.testing.assert_array_equal(relevant_amplitude, distractor_amplitude)

    def test_matched_condition_has_identical_causal_histories(self):
        import numpy as np

        batch = self.boundary.generate_boundary_batch(24, "matched", 19, self.spec)
        for tau in (1.0, 2.0, 8.0):
            trace = self.boundary.prior_trace(batch.inputs, tau)
            for sample in range(len(batch.labels)):
                relevant = np.sort(trace[sample][batch.relevant_mask[sample]])
                distractor = np.sort(trace[sample][batch.distractor_mask[sample]])
                np.testing.assert_array_equal(relevant, distractor)

    def test_fixed_cds_recovers_predicted_boundary(self):
        import torch

        contrasts = {}
        for regime in self.boundary.REGIMES:
            batch = self.boundary.generate_boundary_batch(32, regime, 23, self.spec)
            release = self.boundary._fixed_release(
                batch.inputs, "full", tau=2.0, device=torch.device("cpu")
            )
            contrasts[regime] = self.boundary.transmission_metrics(
                release, batch, tau=2.0
            )["release_contrast"]
        self.assertGreater(contrasts["aligned"], 0.0)
        self.assertAlmostEqual(contrasts["matched"], 0.0, places=7)
        self.assertLess(contrasts["anti_aligned"], 0.0)

    def test_seed_limit_is_enforced(self):
        with self.assertRaises(ValueError):
            self.boundary.parse_int_list("1,2,3,4,5,6")


if __name__ == "__main__":
    unittest.main()
