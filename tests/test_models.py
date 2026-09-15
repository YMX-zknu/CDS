import importlib.util
from pathlib import Path
import sys
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/section_2_6"))


@unittest.skipUnless(HAS_TORCH, "PyTorch is required for model regression checks")
class ModelTests(unittest.TestCase):
    def test_state_reset_gradients_and_ablation_memory(self):
        import torch
        import section_2_6_dvsgesture_train as dvs
        import section_2_6_shd_train as shd
        import section_2_6_stmnist_train as st
        for module in (dvs, shd, st):
            with self.subTest(module=module.__name__):
                layer = module.CDSLayer(3, param_dims=2)
                x = torch.ones(2, 3)
                first = layer(x).detach().clone()
                second = layer(x)
                self.assertTrue(torch.isfinite(second).all())
                self.assertTrue(torch.all(second >= first))
                second.sum().backward()
                self.assertIsNotNone(layer.tau_ca.grad)
                self.assertTrue(torch.isfinite(layer.tau_ca.grad).all())
                layer.reset()
                torch.testing.assert_close(layer(x), first)
                for variant in ("constant_release", "memory_free"):
                    control = module.CDSLayer(3, param_dims=2, variant=variant)
                    one = control(x).detach().clone()
                    torch.testing.assert_close(control(x), one)

    def test_real_perturbation_builder_preserves_original_bins(self):
        import torch
        from section_2_6_dvsgesture_train import add_amplitude_matched_isolated_perturbations
        x = torch.zeros(20, 2, 30)
        x[5:8, :, :5] = 0.5
        result = add_amplitude_matched_isolated_perturbations(x, 0.1, seed=17, guard_steps=4)
        torch.testing.assert_close(result.perturbed[x != 0], x[x != 0])
        self.assertGreater(result.n_added, 0)
        for lag in range(1, 5):
            self.assertFalse(torch.any(result.added_mask[lag:] & (x[:-lag] != 0)))
            self.assertFalse(torch.any(result.added_mask[:-lag] & (x[lag:] != 0)))
            self.assertFalse(torch.any(result.added_mask[lag:] & result.added_mask[:-lag]))


if __name__ == "__main__":
    unittest.main()
