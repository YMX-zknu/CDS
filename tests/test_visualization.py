import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

PATH = Path(__file__).resolve().parents[1] / "experiments/section_2_6/plot_fig6g_cross_modal_inputs.py"
spec = importlib.util.spec_from_file_location("input_visualization", PATH)
visual = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = visual
spec.loader.exec_module(visual)


class VisualizationTests(unittest.TestCase):
    def test_clean_pair_and_temporal_isolation(self):
        args = visual.parse_args(["--dry_run", "--dpi", "150"])
        for dataset in visual.selected_dataset_specs(args):
            with self.subTest(dataset=dataset.key):
                sample = visual.prepare_synthetic_sample(dataset, args)
                clean = visual.without_added_perturbations(sample)
                np.testing.assert_array_equal(clean.perturbed, sample.original)
                self.assertEqual(np.count_nonzero(clean.added_mask), 0)
                x = sample.original.reshape(sample.original.shape[0], -1)
                noise = sample.added_mask.reshape(x.shape)
                np.testing.assert_array_equal(sample.perturbed[sample.original != 0], sample.original[sample.original != 0])
                self.assertGreater(noise.sum(), 0)
                self.assertFalse(np.any(noise & (x != 0)))
                for lag in range(1, 5):
                    self.assertFalse(np.any(noise[lag:] & (x[:-lag] != 0)))
                    self.assertFalse(np.any(noise[:-lag] & (x[lag:] != 0)))
                    self.assertFalse(np.any(noise[lag:] & noise[:-lag]))


if __name__ == "__main__":
    unittest.main()
