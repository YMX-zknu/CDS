import ast
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run


class EntryPointTests(unittest.TestCase):
    def test_all_sources_compile(self):
        for path in (ROOT / "experiments").rglob("*.py"):
            with self.subTest(path=path):
                compile(path.read_text(), str(path), "exec")

    def test_commands_forward_arguments_and_exit_code(self):
        for command, (relative, _) in run.COMMANDS.items():
            self.assertTrue((ROOT / "experiments" / relative).is_file())
            arguments = [] if command in run.MECHANISM_COMMANDS else ["--device", "cpu"]
            with self.subTest(command=command), patch("run.subprocess.run") as launch:
                launch.return_value = subprocess.CompletedProcess([], 7)
                self.assertEqual(run.main([command, *arguments]), 7)
                forwarded = launch.call_args.args[0]
                self.assertEqual(forwarded[2:], arguments)

    def test_ablation_flag_is_opt_in(self):
        spec = importlib.util.spec_from_file_location(
            "launcher26", ROOT / "experiments/section_2_6/run_section_2_6_multiseed.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with patch.object(sys, "argv", ["launcher"]):
            self.assertFalse(module.parse_args().with_ablation)
            self.assertEqual(module.parse_args().seeds, "2026,2027,2028")
        with patch.object(sys, "argv", ["launcher", "--with_ablation"]):
            self.assertTrue(module.parse_args().with_ablation)

    def test_analysis_finds_training_directory(self):
        spec = importlib.util.spec_from_file_location(
            "launcher27", ROOT / "experiments/section_2_7/run_section_2_7_multiseed.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with patch.object(sys, "argv", ["launcher"]):
            root = Path(module.parse_args().section_2_6_code_dir)
            self.assertEqual(module.parse_args().seeds, "2026,2027,2028")
        for dataset in ("dvsgesture", "shd", "stmnist"):
            self.assertTrue((root / f"section_2_6_{dataset}_train.py").is_file())


if __name__ == "__main__":
    unittest.main()
